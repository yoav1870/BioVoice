import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeAlias

import torch


CSV_COLUMNS = ["speaker", "magnitude", "sign_count", "concept_name", "pred_target_idx"]


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


def flatten_scores_rows_generic(
    scores: dict[str, dict[str, dict[str, torch.Tensor]]],
    experimental_sets: Sequence[Sequence[ConceptLike]],
    *,
    entity_label: str,
    entity_value: str,
    target_label: str,
    target_value: int | str,
) -> list[CsvRow]:
    rows: list[CsvRow] = []

    for concept_set_key, layer_scores in scores.items():
        concept_names = _concept_names_for_key(concept_set_key, experimental_sets)
        for score_values in layer_scores.values():
            magnitudes = score_values["magnitude"]
            sign_counts = score_values["sign_count"]
            for concept_name, magnitude, sign_count in zip(
                concept_names, magnitudes, sign_counts
            ):
                rows.append(
                    {
                        entity_label: entity_value,
                        "magnitude": _to_scalar(magnitude),
                        "sign_count": _to_scalar(sign_count),
                        "concept_name": concept_name,
                        target_label: target_value,
                    }
                )

    return rows


def flatten_scores_rows(
    scores: dict[str, dict[str, dict[str, torch.Tensor]]],
    experimental_sets: Sequence[Sequence[ConceptLike]],
    speaker: str,
    pred_target_idx: int,
) -> list[CsvRow]:
    return flatten_scores_rows_generic(
        scores,
        experimental_sets,
        entity_label="speaker",
        entity_value=speaker,
        target_label="pred_target_idx",
        target_value=pred_target_idx,
    )


def write_scores_csv(
    rows: Sequence[CsvRow],
    output_path: Path,
    fieldnames: Sequence[str] = CSV_COLUMNS,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def aggregate_rows_by_weight_generic(
    weighted_rows: Sequence[tuple[Sequence[CsvRow], int]],
    *,
    entity_label: str,
    target_label: str,
) -> list[CsvRow]:
    totals: dict[tuple[str, str, str], dict[str, str | float | int]] = {}
    weights: dict[tuple[str, str, str], int] = {}

    for rows, weight in weighted_rows:
        for row in rows:
            key = (
                str(row[entity_label]),
                str(row["concept_name"]),
                str(row[target_label]),
            )
            if key not in totals:
                totals[key] = {
                    entity_label: row[entity_label],
                    "magnitude": 0.0,
                    "sign_count": 0.0,
                    "concept_name": row["concept_name"],
                    target_label: row[target_label],
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
                entity_label: row[entity_label],
                "magnitude": float(format(float(row["magnitude"]) / total_weight, "g")),
                "sign_count": float(
                    format(float(row["sign_count"]) / total_weight, "g")
                ),
                "concept_name": row["concept_name"],
                target_label: row[target_label],
            }
        )

    aggregated.sort(key=lambda row: (str(row[entity_label]), str(row["concept_name"])))
    return aggregated


def aggregate_rows_by_weight(
    weighted_rows: Sequence[tuple[Sequence[CsvRow], int]],
) -> list[CsvRow]:
    return aggregate_rows_by_weight_generic(
        weighted_rows,
        entity_label="speaker",
        target_label="pred_target_idx",
    )
