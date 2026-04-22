import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeAlias

import torch


CSV_COLUMNS = [
    "speaker",
    "layer",
    "magnitude",
    "sign_count",
    "concept_name",
    "pred_target_idx",
]


class ConceptLike(Protocol):
    name: str


CsvRow: TypeAlias = dict[str, str | float | int]


def _to_scalar(value: torch.Tensor) -> float:
    return float(format(value.item(), "g"))


def _concept_names_for_key(
    concept_set_key: str, experimental_sets: Sequence[Sequence[ConceptLike]]
) -> list[str]:
    experimental_set = experimental_sets[int(concept_set_key.split("-")[0])]
    return [concept.name for concept in experimental_set]


def flatten_scores_rows(
    scores: dict[str, dict[str, dict[str, torch.Tensor]]],
    experimental_sets: Sequence[Sequence[ConceptLike]],
    speaker: str,
    pred_target_idx: int,
) -> list[CsvRow]:
    rows: list[CsvRow] = []

    for concept_set_key, layer_scores in scores.items():
        concept_names = _concept_names_for_key(concept_set_key, experimental_sets)
        for layer_name, score_values in layer_scores.items():
            magnitudes = score_values["magnitude"]
            sign_counts = score_values["sign_count"]
            for concept_name, magnitude, sign_count in zip(
                concept_names, magnitudes, sign_counts
            ):
                rows.append(
                    {
                        "speaker": speaker,
                        "layer": layer_name,
                        "magnitude": _to_scalar(magnitude),
                        "sign_count": _to_scalar(sign_count),
                        "concept_name": concept_name,
                        "pred_target_idx": pred_target_idx,
                    }
                )

    return rows


def write_scores_csv(
    rows: Sequence[CsvRow],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def aggregate_rows_by_weight(
    weighted_rows: Sequence[tuple[Sequence[CsvRow], int]],
) -> list[CsvRow]:
    totals: dict[tuple[str, str, str, int], dict[str, str | float | int]] = {}
    weights: dict[tuple[str, str, str, int], int] = {}

    for rows, weight in weighted_rows:
        for row in rows:
            key = (
                str(row["speaker"]),
                str(row.get("layer", "")),
                str(row["concept_name"]),
                int(row["pred_target_idx"]),
            )
            if key not in totals:
                totals[key] = {
                    "speaker": row["speaker"],
                    "layer": row.get("layer", ""),
                    "magnitude": 0.0,
                    "sign_count": 0.0,
                    "concept_name": row["concept_name"],
                    "pred_target_idx": row["pred_target_idx"],
                }
                weights[key] = 0
            totals[key]["magnitude"] = (
                float(totals[key]["magnitude"]) + float(row["magnitude"]) * weight
            )
            totals[key]["sign_count"] = (
                float(totals[key]["sign_count"]) + float(row["sign_count"]) * weight
            )
            weights[key] += weight

    aggregated: list[CsvRow] = []
    for key, row in totals.items():
        total_weight = weights[key]
        aggregated.append(
            {
                "speaker": row["speaker"],
                "layer": row["layer"],
                "magnitude": float(format(float(row["magnitude"]) / total_weight, "g")),
                "sign_count": float(
                    format(float(row["sign_count"]) / total_weight, "g")
                ),
                "concept_name": row["concept_name"],
                "pred_target_idx": row["pred_target_idx"],
            }
        )

    aggregated.sort(
        key=lambda row: (
            str(row["speaker"]),
            str(row["layer"]),
            str(row["concept_name"]),
        )
    )
    return aggregated
