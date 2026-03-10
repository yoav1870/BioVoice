from __future__ import annotations

from pathlib import Path
from typing import Any

from tcav_core.cav_utils import (
    build_tcav_by_layer,
    compute_cav_accuracies,
    report_cav_quality,
)
from tcav_core.concept_utils import (
    build_concept_datasets,
    build_random_concepts,
    list_positive_concept_dirs,
)
from tcav_core.config import ExperimentConfig
from tcav_core.device_utils import resolve_tcav_device
from tcav_core.model_utils import setup_models
from tcav_core.persistence_utils import save_outputs
from tcav_core.scoring_utils import score_dataset, summarize_rows
from tcav_core.utils import abs_path, project_root, seed_everything


class TCAVRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.tcav_device = resolve_tcav_device(config.tcav_device)

    def run(self) -> None:
        seed_everything(self.config.seed)
        paths = self._resolve_and_validate_paths()

        (
            redim_model,
            wrapped_model,
            speaker_to_id,
            frame_normalizer,
            n_mels,
            target_frames,
        ) = setup_models(self.config, self.tcav_device)

        concept_dirs = list_positive_concept_dirs(
            concept_root=paths["concept_root"],
            excluded_names=self.config.excluded_concept_names,
        )
        print(f"Concepts: {[d.name for d in concept_dirs]}")

        loader_kwargs = self._loader_kwargs()
        positive_concepts = build_concept_datasets(
            concept_dirs=concept_dirs,
            n_mels=n_mels,
            frame_normalizer=frame_normalizer,
            config=self.config,
            loader_kwargs=loader_kwargs,
            tcav_device=self.tcav_device,
        )
        random_concepts = build_random_concepts(
            config=self.config,
            n_mels=n_mels,
            target_frames=target_frames,
            frame_normalizer=frame_normalizer,
            spec_fn=redim_model.spec,
            random_source_csv=paths["random_source_csv"],
            loader_kwargs=loader_kwargs,
            tcav_device=self.tcav_device,
        )

        tcav_by_layer = build_tcav_by_layer(
            wrapped_model=wrapped_model,
            config=self.config,
            cav_save_path=self.config.cav_save_path,
            model_id=self.config.model_id,
        )

        cav_acc_df = compute_cav_accuracies(
            tcav_by_layer=tcav_by_layer,
            positive_concepts=positive_concepts,
            random_concepts=random_concepts,
            config=self.config,
            tcav_device=self.tcav_device,
        )
        report_cav_quality(
            cav_acc_df=cav_acc_df,
            config=self.config,
            cav_save_path=self.config.cav_save_path,
        )

        rows_df = score_dataset(
            dataset_csv=paths["dataset_csv"],
            config=self.config,
            speaker_to_id=speaker_to_id,
            redim_model=redim_model,
            wrapped_model=wrapped_model,
            frame_normalizer=frame_normalizer,
            tcav_device=self.tcav_device,
            tcav_by_layer=tcav_by_layer,
            positive_concepts=positive_concepts,
            random_concepts=random_concepts,
        )
        summary_df, effect_summary_df = summarize_rows(rows_df)

        save_outputs(
            output_dir=paths["output_dir"],
            config=self.config,
            rows_df=rows_df,
            summary_df=summary_df,
            effect_summary_df=effect_summary_df,
            cav_acc_df=cav_acc_df,
            run_metadata={
                "dataset_csv_path": str(paths["dataset_csv"]),
                "output_dir": str(paths["output_dir"]),
                "concept_root": str(paths["concept_root"]),
                "head_path": str(paths["head_path"]),
                "random_source_csv_path": str(paths["random_source_csv"]),
                "n_mels": int(n_mels),
                "target_frames_effective": int(target_frames),
                "tcav_device": str(self.tcav_device),
                "rows_scored": int(len(rows_df)),
            },
        )

    def _resolve_and_validate_paths(self) -> dict[str, Path]:
        dataset_csv = abs_path(self.config.dataset_csv_path)
        output_dir = abs_path(self.config.output_dir)
        concept_root = abs_path(self.config.concept_root)
        head_path = abs_path(self.config.head_path)

        random_source_csv = (
            abs_path(self.config.random_source_csv_path)
            if self.config.random_source_csv_path
            else dataset_csv
        )

        if self.config.cav_save_path is not None:
            self.config.cav_save_path = abs_path(self.config.cav_save_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.cav_save_path is not None:
            self.config.cav_save_path.mkdir(parents=True, exist_ok=True)

        if not dataset_csv.exists():
            raise FileNotFoundError(f"Missing dataset csv: {dataset_csv}")
        if not concept_root.exists():
            raise FileNotFoundError(f"Missing concept root: {concept_root}")
        if not head_path.exists():
            raise FileNotFoundError(f"Missing head checkpoint: {head_path}")
        if not random_source_csv.exists():
            raise FileNotFoundError(f"Missing random source csv: {random_source_csv}")

        print(f"PROJECT_ROOT: {project_root()}")
        print(f"DATASET_CSV_PATH: {dataset_csv}")
        print(f"OUTPUT_DIR: {output_dir}")
        print(f"CONCEPT_ROOT: {concept_root}")
        print(f"NEGATIVE_MODE: {self.config.negative_mode}")
        print(f"TARGET_MODE: {self.config.target_mode}")
        print(f"TCAV_DEVICE: {self.tcav_device}")
        print(
            f"CAV_CACHE: {self.config.cav_save_path} "
            f"(force_train_cavs={self.config.force_train_cavs})"
        )

        return {
            "dataset_csv": dataset_csv,
            "output_dir": output_dir,
            "concept_root": concept_root,
            "head_path": head_path,
            "random_source_csv": random_source_csv,
        }

    def _loader_kwargs(self) -> dict[str, Any]:
        workers = max(0, int(self.config.data_loader_num_workers))
        pin_memory = bool(
            self.config.data_loader_pin_memory and self.tcav_device.type == "cuda"
        )
        kwargs: dict[str, Any] = {
            "batch_size": int(self.config.batch_size),
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": pin_memory,
        }
        if workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs
