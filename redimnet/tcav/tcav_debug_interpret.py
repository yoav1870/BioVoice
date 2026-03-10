from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from tcav_core.cav_utils import build_tcav_by_layer, compute_cavs_with_context
from tcav_core.concept_utils import (
    build_concept_datasets,
    build_random_concepts,
    list_positive_concept_dirs,
)
from tcav_core.config import CONFIG, ExperimentConfig
from tcav_core.device_utils import resolve_tcav_device
from tcav_core.model_utils import setup_models
from tcav_core.scoring_utils import (
    build_experimental_sets,
    metric_to_list,
    predict_label_and_target,
    resolve_target_idx,
    wav_path_to_mel4d,
)
from tcav_core.utils import abs_path, seed_everything


def loader_kwargs(config: ExperimentConfig, tcav_device) -> dict[str, Any]:
    workers = max(0, int(config.data_loader_num_workers))
    pin_memory = bool(config.data_loader_pin_memory and tcav_device.type == "cuda")
    kwargs: dict[str, Any] = {
        "batch_size": int(config.batch_size),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def resolve_paths(config: ExperimentConfig) -> dict[str, Path]:
    dataset_csv = abs_path(config.dataset_csv_path)
    concept_root = abs_path(config.concept_root)
    head_path = abs_path(config.head_path)
    random_source_csv = (
        abs_path(config.random_source_csv_path)
        if config.random_source_csv_path
        else dataset_csv
    )
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Missing dataset csv: {dataset_csv}")
    if not concept_root.exists():
        raise FileNotFoundError(f"Missing concept root: {concept_root}")
    if not head_path.exists():
        raise FileNotFoundError(f"Missing head checkpoint: {head_path}")
    if not random_source_csv.exists():
        raise FileNotFoundError(f"Missing random source csv: {random_source_csv}")

    return {
        "dataset_csv": dataset_csv,
        "concept_root": concept_root,
        "head_path": head_path,
        "random_source_csv": random_source_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick TCAV interpret debugger: print raw vectors.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--layer-key", type=str, default="")
    parser.add_argument("--max-concepts", type=int, default=3)
    parser.add_argument("--max-random", type=int, default=1)
    parser.add_argument("--max-experiments", type=int, default=20)
    parser.add_argument("--concept-name", type=str, default="")
    parser.add_argument("--force-train-cavs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIG

    seed_everything(config.seed)
    tcav_device = resolve_tcav_device(config.tcav_device)
    paths = resolve_paths(config)

    (
        redim_model,
        wrapped_model,
        speaker_to_id,
        frame_normalizer,
        n_mels,
        target_frames,
    ) = setup_models(config, tcav_device)

    concept_dirs = list_positive_concept_dirs(
        concept_root=paths["concept_root"],
        excluded_names=config.excluded_concept_names,
    )

    kwargs = loader_kwargs(config, tcav_device)
    positive_concepts = build_concept_datasets(
        concept_dirs=concept_dirs,
        n_mels=n_mels,
        frame_normalizer=frame_normalizer,
        config=config,
        loader_kwargs=kwargs,
        tcav_device=tcav_device,
    )
    random_concepts = build_random_concepts(
        config=config,
        n_mels=n_mels,
        target_frames=target_frames,
        frame_normalizer=frame_normalizer,
        spec_fn=redim_model.spec,
        random_source_csv=paths["random_source_csv"],
        loader_kwargs=kwargs,
        tcav_device=tcav_device,
    )

    if args.concept_name:
        positive_concepts = [
            concept
            for concept in positive_concepts
            if concept.name == args.concept_name
        ]
        if not positive_concepts:
            raise RuntimeError(
                f"No concept matched --concept-name={args.concept_name!r}"
            )

    if args.max_concepts > 0:
        positive_concepts = positive_concepts[: args.max_concepts]
    if args.max_random > 0:
        random_concepts = random_concepts[: args.max_random]

    tcav_by_layer = build_tcav_by_layer(
        wrapped_model=wrapped_model,
        config=config,
        cav_save_path=config.cav_save_path,
        model_id=config.model_id,
    )

    if args.layer_key:
        if args.layer_key not in tcav_by_layer:
            raise RuntimeError(
                f"Unknown --layer-key={args.layer_key!r}. "
                f"Available: {list(tcav_by_layer.keys())}"
            )
        layer_key = args.layer_key
    else:
        layer_key = next(iter(tcav_by_layer.keys()))

    tcav = tcav_by_layer[layer_key]
    experimental_sets = build_experimental_sets(positive_concepts, random_concepts)

    force_train = bool(config.force_train_cavs or args.force_train_cavs)
    print(f"Computing/loading CAVs (force_train={force_train})")
    _ = compute_cavs_with_context(
        tcav=tcav,
        experimental_sets=experimental_sets,
        force_train=force_train,
        force_av_load_cpu=config.force_av_load_cpu,
    )

    dataset_df = pd.read_csv(paths["dataset_csv"])
    if dataset_df.empty:
        raise RuntimeError("Dataset CSV is empty.")

    if args.sample_index < 0 or args.sample_index >= len(dataset_df):
        raise RuntimeError(
            f"sample-index out of range: {args.sample_index}. "
            f"valid range: 0..{len(dataset_df)-1}"
        )

    row = dataset_df.iloc[args.sample_index]
    wav_path = Path(str(row[config.path_column]))
    true_label = str(row[config.label_column])
    if not wav_path.exists():
        raise FileNotFoundError(f"Sample path does not exist: {wav_path}")

    mel_4d = wav_path_to_mel4d(
        wav_path=wav_path,
        redim_model=redim_model,
        frame_normalizer=frame_normalizer,
        seed=config.seed,
        tcav_device=tcav_device,
    )
    pred_label, pred_prob, pred_target_idx = predict_label_and_target(
        wrapped_model, mel_4d
    )
    true_target_idx = speaker_to_id.get(true_label)
    target_idx, target_kind = resolve_target_idx(
        config, true_target_idx, pred_target_idx
    )

    scores = tcav.interpret(
        inputs=mel_4d,
        experimental_sets=experimental_sets,
        target=target_idx,
    )

    concept_name_by_id = {
        concept.id: concept.name for concept in (positive_concepts + random_concepts)
    }

    print("\n=== DEBUG CONTEXT ===")
    print(f"sample_index: {args.sample_index}")
    print(f"wav_path: {wav_path}")
    print(f"true_label: {true_label}")
    print(f"pred_label: {pred_label} (p={pred_prob:.6f})")
    print(f"target_idx: {target_idx} (target_kind={target_kind})")
    print(f"layer_key: {layer_key}")
    print(f"num_experimental_sets: {len(experimental_sets)}")

    printed = 0
    print("\n=== RAW TCAV.interpret VECTORS ===")
    for exp_key, layer_map in scores.items():
        if printed >= args.max_experiments:
            break
        try:
            concept_id_str, random_id_str = str(exp_key).split("-", maxsplit=1)
            concept_id = int(concept_id_str)
            random_id = int(random_id_str)
            concept_name = concept_name_by_id.get(concept_id, f"id_{concept_id}")
            random_name = concept_name_by_id.get(random_id, f"id_{random_id}")
        except Exception:
            concept_name = "unknown"
            random_name = "unknown"

        metrics = next(iter(layer_map.values()))
        sign_count = metric_to_list(metrics["sign_count"])
        magnitude = metric_to_list(metrics["magnitude"])

        print(
            f"[{printed+1}] exp_key={exp_key} "
            f"concept={concept_name} random={random_name}"
        )
        print(f"    sign_count={sign_count}")
        print(f"    magnitude={magnitude}")
        printed += 1


if __name__ == "__main__":
    main()
