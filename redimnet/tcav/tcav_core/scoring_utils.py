from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from captum.concept import Concept, TCAV

from tcav_core.config import ExperimentConfig
from tcav_core.frame import FrameNormalizer
from tcav_core.modeling import ReDimNetMelLogitsWrapper
from tcav_core.utils import stable_int_seed


def build_experimental_sets(
    positive_concepts: list[Concept],
    random_concepts: list[Concept],
) -> list[list[Concept]]:
    return [
        [concept, random_concept]
        for concept in positive_concepts
        for random_concept in random_concepts
    ]


def metric_to_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def wav_path_to_mel4d(
    wav_path: Path,
    redim_model: Any,
    frame_normalizer: FrameNormalizer,
    seed: int,
    tcav_device: torch.device,
) -> torch.Tensor:
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav[:1, :].float().to(tcav_device)
    with torch.no_grad():
        mel = redim_model.spec(wav)
    rng = np.random.default_rng(stable_int_seed(wav_path.as_posix(), seed))
    mel = frame_normalizer.normalize(mel, rng)
    return mel.unsqueeze(0)


def predict_label_and_target(
    wrapped_model: ReDimNetMelLogitsWrapper,
    mel_4d: torch.Tensor,
) -> tuple[str, float, int]:
    with torch.no_grad():
        logits = wrapped_model(mel_4d)
        probs = F.softmax(logits, dim=1)[0]
        pred_target_idx = int(torch.argmax(probs).item())
        pred_label = str(pred_target_idx)
        pred_prob = float(probs[pred_target_idx].item())
    return pred_label, pred_prob, pred_target_idx


def resolve_target_idx(
    config: ExperimentConfig,
    true_target_idx: Optional[int],
    pred_target_idx: int,
) -> tuple[int, str]:
    mode = config.target_mode.strip().lower()
    if mode == "true_label":
        if true_target_idx is None:
            raise RuntimeError(
                "true_label target mode requires known true label mapping."
            )
        return int(true_target_idx), "true_label"
    if mode == "predicted":
        return int(pred_target_idx), "predicted"
    if mode == "fixed":
        if config.fixed_target_idx is None:
            raise RuntimeError("fixed target mode requires fixed_target_idx.")
        return int(config.fixed_target_idx), "fixed"
    raise ValueError("target_mode must be 'true_label', 'predicted', or 'fixed'.")


def extract_interpret_rows(
    *,
    wav_path: Path,
    true_label: str,
    pred_label: str,
    pred_prob: float,
    layer_key: str,
    scores: dict[str, dict[str, dict[str, Any]]],
    concept_name_by_id: dict[int, str],
    target_idx: int,
    target_kind: str,
    effect_epsilon: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_key, layer_scores in scores.items():
        try:
            concept_id_str, random_id_str = str(exp_key).split("-", maxsplit=1)
            concept_id = int(concept_id_str)
            random_id = int(random_id_str)
        except Exception:
            continue

        if concept_id not in concept_name_by_id or random_id not in concept_name_by_id:
            continue

        concept_name = concept_name_by_id[concept_id]
        random_name = concept_name_by_id[random_id]

        for layer_name, metrics in layer_scores.items():
            sign_count = metrics.get("sign_count")
            magnitude = metrics.get("magnitude")
            if sign_count is None or magnitude is None:
                continue

            sign_count_list = metric_to_list(sign_count)
            magnitude_list = metric_to_list(magnitude)
            if len(sign_count_list) < 2 or len(magnitude_list) < 2:
                continue

            pos_concept = float(sign_count_list[0])
            pos_random = float(sign_count_list[1])
            mag_concept = float(magnitude_list[0])
            mag_random = float(magnitude_list[1])

            pos_delta = pos_concept - pos_random
            mag_delta = mag_concept - mag_random
            mag_ratio = mag_delta / (abs(mag_random) + float(effect_epsilon))
            mag_abs_ratio = abs(mag_delta) / (abs(mag_random) + float(effect_epsilon))

            rows.append(
                {
                    "path": str(wav_path),
                    "layer_key": layer_key,
                    "concept name": concept_name,
                    "random concept": random_name,
                    "layer name": layer_name,
                    "pos": pos_concept,
                    "mag": mag_concept,
                    "pos_random": pos_random,
                    "mag_random": mag_random,
                    "pos_effect_delta": pos_delta,
                    "mag_effect_delta": mag_delta,
                    "mag_effect_ratio": mag_ratio,
                    "mag_effect_abs_ratio": mag_abs_ratio,
                    "target_idx": int(target_idx),
                    "target_kind": target_kind,
                    "true label": true_label,
                    "predicted label": pred_label,
                    "predicted probability": float(pred_prob),
                }
            )
    return rows


def score_dataset(
    *,
    dataset_csv: Path,
    config: ExperimentConfig,
    speaker_to_id: dict[str, int],
    redim_model: Any,
    wrapped_model: ReDimNetMelLogitsWrapper,
    frame_normalizer: FrameNormalizer,
    tcav_device: torch.device,
    tcav_by_layer: dict[str, TCAV],
    positive_concepts: list[Concept],
    random_concepts: list[Concept],
) -> pd.DataFrame:
    dataset_df = pd.read_csv(dataset_csv)
    required_cols = [config.path_column, config.label_column]
    missing = [c for c in required_cols if c not in dataset_df.columns]
    if missing:
        raise RuntimeError(
            f"Dataset csv missing columns {missing}. Got: {list(dataset_df.columns)}"
        )

    experimental_sets = build_experimental_sets(positive_concepts, random_concepts)
    concept_name_by_id = {
        concept.id: concept.name for concept in (positive_concepts + random_concepts)
    }

    rows: list[dict[str, Any]] = []
    skipped_missing_path = 0
    skipped_unknown_label = 0

    for _, row in dataset_df.iterrows():
        wav_path = Path(str(row[config.path_column]))
        true_label = str(row[config.label_column])

        if not wav_path.exists():
            skipped_missing_path += 1
            continue

        true_target_idx = speaker_to_id.get(true_label)
        if config.target_mode == "true_label" and true_target_idx is None:
            skipped_unknown_label += 1
            continue

        mel_4d = wav_path_to_mel4d(
            wav_path=wav_path,
            redim_model=redim_model,
            frame_normalizer=frame_normalizer,
            seed=config.seed,
            tcav_device=tcav_device,
        )

        pred_label, pred_prob, pred_target_idx = predict_label_and_target(
            wrapped_model=wrapped_model,
            mel_4d=mel_4d,
        )
        target_idx, target_kind = resolve_target_idx(
            config=config,
            true_target_idx=true_target_idx,
            pred_target_idx=pred_target_idx,
        )

        for layer_key, tcav in tcav_by_layer.items():
            scores = tcav.interpret(
                inputs=mel_4d,
                experimental_sets=experimental_sets,
                target=target_idx,
            )
            rows.extend(
                extract_interpret_rows(
                    wav_path=wav_path,
                    true_label=true_label,
                    pred_label=pred_label,
                    pred_prob=pred_prob,
                    layer_key=layer_key,
                    scores=scores,
                    concept_name_by_id=concept_name_by_id,
                    target_idx=target_idx,
                    target_kind=target_kind,
                    effect_epsilon=config.effect_epsilon,
                )
            )

    rows_df = pd.DataFrame(
        rows,
        columns=[
            "path",
            "layer_key",
            "concept name",
            "random concept",
            "layer name",
            "pos",
            "mag",
            "pos_random",
            "mag_random",
            "pos_effect_delta",
            "mag_effect_delta",
            "mag_effect_ratio",
            "mag_effect_abs_ratio",
            "target_idx",
            "target_kind",
            "true label",
            "predicted label",
            "predicted probability",
        ],
    )

    print(f"Dataset rows: {len(dataset_df)}")
    print(f"Rows scored: {len(rows_df)}")
    print(f"Skipped missing path: {skipped_missing_path}")
    print(f"Skipped unknown label: {skipped_unknown_label}")
    return rows_df


def summarize_rows(rows_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "concept name",
                "random concept",
                "layer_key",
                "layer name",
                "target_kind",
                "target_idx",
                "pos_mean",
                "pos_std",
                "pos_median",
                "mag_mean",
                "mag_std",
                "mag_median",
                "pos_effect_delta_mean",
                "mag_effect_delta_mean",
                "mag_effect_ratio_mean",
                "mag_effect_abs_ratio_mean",
                "n",
            ]
        )
        effect_summary_df = pd.DataFrame(
            columns=[
                "concept name",
                "layer_key",
                "layer name",
                "target_kind",
                "target_idx",
                "pos_effect_delta_mean",
                "pos_effect_delta_std",
                "mag_effect_delta_mean",
                "mag_effect_delta_std",
                "mag_effect_ratio_mean",
                "mag_effect_ratio_std",
                "mag_effect_abs_ratio_mean",
                "mag_effect_abs_ratio_std",
                "mag_effect_positive_rate",
                "n",
            ]
        )
        return summary_df, effect_summary_df

    summary_df = (
        rows_df.groupby(
            [
                "concept name",
                "random concept",
                "layer_key",
                "layer name",
                "target_kind",
                "target_idx",
            ],
            as_index=False,
        )
        .agg(
            pos_mean=("pos", "mean"),
            pos_std=("pos", "std"),
            pos_median=("pos", "median"),
            mag_mean=("mag", "mean"),
            mag_std=("mag", "std"),
            mag_median=("mag", "median"),
            pos_effect_delta_mean=("pos_effect_delta", "mean"),
            mag_effect_delta_mean=("mag_effect_delta", "mean"),
            mag_effect_ratio_mean=("mag_effect_ratio", "mean"),
            mag_effect_abs_ratio_mean=("mag_effect_abs_ratio", "mean"),
            n=("path", "count"),
        )
        .sort_values(["layer_key", "layer name", "concept name", "random concept"])
    )

    effect_summary_df = (
        rows_df.groupby(
            ["concept name", "layer_key", "layer name", "target_kind", "target_idx"],
            as_index=False,
        )
        .agg(
            pos_effect_delta_mean=("pos_effect_delta", "mean"),
            pos_effect_delta_std=("pos_effect_delta", "std"),
            mag_effect_delta_mean=("mag_effect_delta", "mean"),
            mag_effect_delta_std=("mag_effect_delta", "std"),
            mag_effect_ratio_mean=("mag_effect_ratio", "mean"),
            mag_effect_ratio_std=("mag_effect_ratio", "std"),
            mag_effect_abs_ratio_mean=("mag_effect_abs_ratio", "mean"),
            mag_effect_abs_ratio_std=("mag_effect_abs_ratio", "std"),
            mag_effect_positive_rate=(
                "mag_effect_delta",
                lambda s: float((s > 0).mean()),
            ),
            n=("path", "count"),
        )
        .sort_values(["layer_key", "layer name", "concept name"])
    )

    return summary_df, effect_summary_df
