"""CAV computation and TCAV operations."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from captum.concept import Concept, TCAV
from captum.concept._core.cav import CAV

from tcav_core.config import ExperimentConfig
from tcav_core.modeling import ReDimNetMelLogitsWrapper, module_name_in_model


def build_tcav_by_layer(
    wrapped_model: ReDimNetMelLogitsWrapper,
    config: ExperimentConfig,
    cav_save_path: Path | None,
    model_id: str,
) -> dict[str, TCAV]:
    """Build TCAV objects for each target layer."""
    from tcav_core.modeling import build_target_layers

    all_layer_modules = build_target_layers(wrapped_model)
    missing = [k for k in config.target_layer_keys if k not in all_layer_modules]
    if missing:
        raise ValueError(
            f"Unknown layer keys: {missing}. Available: {list(all_layer_modules.keys())}"
        )

    tcav_by_layer: dict[str, TCAV] = {}
    for layer_key in config.target_layer_keys:
        layer_module = all_layer_modules[layer_key]
        layer_name = module_name_in_model(wrapped_model, layer_module)

        # Ensure CAV save directory exists
        if cav_save_path:
            cav_save_path.mkdir(parents=True, exist_ok=True)

        tcav_by_layer[layer_key] = TCAV(
            wrapped_model,
            [layer_name],
            test_split_ratio=float(config.cav_test_split_ratio),
            save_path=str(cav_save_path) if cav_save_path else None,
            model_id=model_id,
        )
        print(
            f"TCAV ready: {layer_key} -> {layer_name} "
            f"(caching to: {cav_save_path or 'disabled'})"
        )

    return tcav_by_layer


@contextlib.contextmanager
def _cpu_torch_load_context(force_cpu: bool):
    """Context manager to force torch.load to use CPU."""
    if not force_cpu:
        yield
        return

    original_torch_load = torch.load

    def _torch_load_cpu(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_cpu  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]


def compute_cavs_with_context(
    *,
    tcav: TCAV,
    experimental_sets: list[list[Concept]],
    force_train: bool,
    force_av_load_cpu: bool,
) -> dict[str, dict[str, CAV]]:
    with _cpu_torch_load_context(force_av_load_cpu):
        return tcav.compute_cavs(
            experimental_sets,
            force_train=force_train,
        )


def compute_cav_accuracies(
    tcav_by_layer: dict[str, TCAV],
    positive_concepts: list[Concept],
    random_concepts: list[Concept],
    config: ExperimentConfig,
    tcav_device: torch.device,
) -> pd.DataFrame:
    """Compute CAVs and extract accuracy metrics."""
    all_rows: list[dict[str, Any]] = []
    concept_name_by_id = {
        concept.id: concept.name for concept in (positive_concepts + random_concepts)
    }
    experimental_sets = [
        [concept, random_concept]
        for concept in positive_concepts
        for random_concept in random_concepts
    ]

    for layer_key, tcav in tcav_by_layer.items():
        print(f"Computing CAVs: {layer_key}")
        cavs_dict = compute_cavs_with_context(
            tcav=tcav,
            experimental_sets=experimental_sets,
            force_train=config.force_train_cavs,
            force_av_load_cpu=config.force_av_load_cpu,
        )

        for concepts_key, layer_map in cavs_dict.items():
            try:
                concept_id_str, random_id_str = str(concepts_key).split("-", maxsplit=1)
                concept_id = int(concept_id_str)
                random_id = int(random_id_str)
            except Exception:
                continue

            if (
                concept_id not in concept_name_by_id
                or random_id not in concept_name_by_id
            ):
                continue

            concept_name = concept_name_by_id[concept_id]
            random_name = concept_name_by_id[random_id]

            for layer_name, cav_obj in layer_map.items():
                if cav_obj is None or cav_obj.stats is None:
                    continue
                acc = cav_obj.stats.get("accs", cav_obj.stats.get("acc", None))
                if isinstance(acc, torch.Tensor):
                    acc = acc.detach().cpu().item()
                all_rows.append(
                    {
                        "layer_key": layer_key,
                        "concept name": concept_name,
                        "random concept": random_name,
                        "layer name": layer_name,
                        "cav acc": float(acc) if acc is not None else float("nan"),
                    }
                )

        if tcav_device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(
        all_rows,
        columns=[
            "layer_key",
            "concept name",
            "random concept",
            "layer name",
            "cav acc",
        ],
    )


def report_cav_quality(
    cav_acc_df: pd.DataFrame, config: ExperimentConfig, cav_save_path: Path | None
) -> None:
    """Print CAV quality statistics and warnings."""
    if cav_acc_df.empty or "cav acc" not in cav_acc_df.columns:
        print("WARNING: empty CAV accuracy dataframe.")
        return

    scores = cav_acc_df["cav acc"].dropna()
    if scores.empty:
        print("WARNING: CAV accuracies are all NaN.")
        return

    # Report CAV cache status
    if cav_save_path:
        cav_files = list(cav_save_path.glob("*.pkl"))
        print(f"CAV cache location: {cav_save_path}")
        print(f"Cached CAV files: {len(cav_files)}")
        if cav_files:
            print(f"  (delete these files to force retraining)")
    else:
        print("WARNING: CAV caching disabled (no save_path configured)")

    print(
        f"CAV quality stats: mean={scores.mean():.4f} median={scores.median():.4f} "
        f"min={scores.min():.4f} max={scores.max():.4f}"
    )

    mean_acc = float(scores.mean())
    threshold = float(config.cav_quality_warn_threshold)
    if mean_acc < threshold:
        print(
            f"WARNING: mean CAV accuracy below threshold. "
            f"mean={mean_acc:.4f} threshold={threshold:.4f}. "
            "Consider cleaner concepts, more samples, better layer, and better random baselines."
        )

    concept_mean = cav_acc_df.groupby("concept name")["cav acc"].mean()
    worst = concept_mean.nsmallest(5).reset_index(name="cav_acc_mean")
    best = concept_mean.nlargest(5).reset_index(name="cav_acc_mean")
    print("Lowest concept CAV accuracies:")
    print(worst.to_string(index=False))
    print("Highest concept CAV accuracies:")
    print(best.to_string(index=False))
