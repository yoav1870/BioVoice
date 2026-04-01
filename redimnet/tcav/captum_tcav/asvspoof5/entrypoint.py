# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
TEMP_DIR = RUNTIME_DIR / "tmp"
CAV_SAVE_PATH = RUNTIME_DIR / "cav_cache"
OUTPUT_ROOT = RUNTIME_DIR / "output"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
CAV_SAVE_PATH.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)

from captum.concept import Concept, TCAV

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from captum_tcav.common import chunk_paths, resolve_layers
    from captum_tcav.concepts import make_iter
    from captum_tcav.export_csv import (
        CsvRow,
        aggregate_rows_by_weight_generic,
        flatten_scores_rows_generic,
        write_scores_csv,
    )
    from captum_tcav.asvspoof5.config import load_config
    from captum_tcav.asvspoof5.data import (
        build_tar_index,
        fixed_speakers_for_partition,
        load_input,
        load_manifest,
        partition_for_system,
        select_subset,
        validate_config,
    )
    from captum_tcav.asvspoof5.modeling import load_model
else:
    from ..common import chunk_paths, resolve_layers
    from ..concepts import make_iter
    from ..export_csv import (
        CsvRow,
        aggregate_rows_by_weight_generic,
        flatten_scores_rows_generic,
        write_scores_csv,
    )
    from .config import load_config
    from .data import (
        build_tar_index,
        fixed_speakers_for_partition,
        load_input,
        load_manifest,
        partition_for_system,
        select_subset,
        validate_config,
    )
    from .modeling import load_model


config = load_config()
validate_config(config)
OUTPUT_DIR = OUTPUT_ROOT / config.output_subdir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

concepts = [
    Concept(
        idx,
        concept_name,
        make_iter(
            concept_root=config.concept_root,
            concept_names=config.concept_names,
            random_concept_name=config.random_concept_name,
            random_seed=config.random_seed,
            name=concept_name,
        ),
    )
    for idx, concept_name in enumerate(config.concept_names)
]
random_data_iter = make_iter(
    concept_root=config.concept_root,
    concept_names=config.concept_names,
    random_concept_name=config.random_concept_name,
    random_seed=config.random_seed,
    name=config.random_concept_name,
)
random_concept = Concept(len(concepts), config.random_concept_name, random_data_iter)
experimental_sets = [[concept, random_concept] for concept in concepts]

for system_id in config.system_ids:
    source_partition = partition_for_system(system_id)
    print(f"=== Running TCAV for {system_id} ({source_partition}) ===")

    manifest = load_manifest(config, system_id)
    subset = select_subset(config, system_id, manifest)
    tar_index = build_tar_index(config, system_id)
    model = load_model(config, system_id)
    resolved_layers = resolve_layers(model, config.layers)
    mytcav = TCAV(
        model=model,
        layers=resolved_layers,
        save_path=str(CAV_SAVE_PATH),
    )

    subset_rows = subset.selected_rows.copy().reset_index(drop=True)
    csv_rows: list[CsvRow] = []

    for speaker_id in subset.selected_speakers:
        speaker_df = subset_rows[subset_rows["speaker_id"].eq(speaker_id)].copy()
        utt_ids = speaker_df["utt_id"].astype(str).tolist()
        utt_chunks = (
            [[utt_id] for utt_id in utt_ids]
            if config.output_mode == "row"
            else chunk_paths(utt_ids, config.max_clips_per_chunk)
        )
        print(
            f"Processing system={system_id} speaker={speaker_id} "
            f"chunks={len(utt_chunks)} utts={len(utt_ids)}"
        )
        weighted_chunk_rows: list[tuple[list[CsvRow], int]] = []
        for utt_chunk in utt_chunks:
            inputs = load_input(config, model, utt_chunk, tar_index)
            scores = mytcav.interpret(inputs, experimental_sets, target=0)
            flattened_rows = flatten_scores_rows_generic(
                scores,
                experimental_sets,
                entity_label="speaker_id",
                entity_value=speaker_id,
                target_label="target_class",
                target_value=config.example_class,
            )
            if config.output_mode == "row":
                utt_id = str(utt_chunk[0])
                for row in flattened_rows:
                    concept_name = str(row["concept_name"])
                    if concept_name == config.random_concept_name:
                        continue
                    csv_rows.append(
                        {
                            "system_id": system_id,
                            "source_partition": source_partition,
                            "split": config.split_name,
                            "speaker_id": speaker_id,
                            "utt_id": utt_id,
                            "magnitude": row["magnitude"],
                            "sign_count": row["sign_count"],
                            "concept_name": concept_name,
                            "target_class": row["target_class"],
                        }
                    )
            else:
                weighted_chunk_rows.append((flattened_rows, len(utt_chunk)))

        if config.output_mode == "mean":
            per_speaker_rows = aggregate_rows_by_weight_generic(
                weighted_chunk_rows,
                entity_label="speaker_id",
                target_label="target_class",
            )
            for row in per_speaker_rows:
                concept_name = str(row["concept_name"])
                if concept_name == config.random_concept_name:
                    continue
                csv_rows.append(
                    {
                        "system_id": system_id,
                        "source_partition": source_partition,
                        "split": config.split_name,
                        "speaker_id": row["speaker_id"],
                        "magnitude": row["magnitude"],
                        "sign_count": row["sign_count"],
                        "concept_name": concept_name,
                        "target_class": row["target_class"],
                    }
                )

    if config.output_mode == "mean":
        system_totals: dict[str, dict[str, float | str]] = {}
        for row in csv_rows:
            concept_name = str(row["concept_name"])
            if concept_name not in system_totals:
                system_totals[concept_name] = {
                    "system_id": system_id,
                    "source_partition": source_partition,
                    "split": config.split_name,
                    "magnitude": 0.0,
                    "sign_count": 0.0,
                    "concept_name": concept_name,
                    "target_class": config.example_class,
                }
            system_totals[concept_name]["magnitude"] = (
                float(system_totals[concept_name]["magnitude"]) + float(row["magnitude"])
            )
            system_totals[concept_name]["sign_count"] = (
                float(system_totals[concept_name]["sign_count"]) + float(row["sign_count"])
            )

        system_rows: list[CsvRow] = []
        num_speakers = len(subset.selected_speakers)
        for concept_name, row in system_totals.items():
            system_rows.append(
                {
                    "system_id": row["system_id"],
                    "source_partition": row["source_partition"],
                    "split": row["split"],
                    "speaker_id": "ALL_SELECTED",
                    "magnitude": float(format(float(row["magnitude"]) / num_speakers, "g")),
                    "sign_count": float(format(float(row["sign_count"]) / num_speakers, "g")),
                    "concept_name": concept_name,
                    "target_class": row["target_class"],
                }
            )
        csv_rows = sorted(system_rows, key=lambda row: str(row["concept_name"]))
        fieldnames = [
            "system_id",
            "source_partition",
            "split",
            "speaker_id",
            "magnitude",
            "sign_count",
            "concept_name",
            "target_class",
        ]
    else:
        csv_rows = sorted(
            csv_rows,
            key=lambda row: (str(row["speaker_id"]), str(row["utt_id"]), str(row["concept_name"])),
        )
        fieldnames = [
            "system_id",
            "source_partition",
            "split",
            "speaker_id",
            "utt_id",
            "magnitude",
            "sign_count",
            "concept_name",
            "target_class",
        ]

    csv_path = write_scores_csv(
        rows=csv_rows,
        output_path=OUTPUT_DIR / f"tcav_{source_partition}_{system_id}_{config.example_class}.csv",
        fieldnames=fieldnames,
    )

    subset_summary = {
        "system_id": system_id,
        "source_partition": source_partition,
        "split": config.split_name,
        "target_class": config.example_class,
        "subset_seed": config.subset_seed,
        "subset_num_speakers": config.subset_num_speakers,
        "subset_utts_per_speaker": config.subset_utts_per_speaker,
        "selected_speakers": subset.selected_speakers,
        "fixed_speakers": fixed_speakers_for_partition(config, source_partition),
        "excluded_speakers": (
            config.excluded_train_speakers
            if source_partition == "train"
            else config.excluded_dev_speakers
        ),
        "selected_rows": int(len(subset_rows)),
    }
    (OUTPUT_DIR / f"subset_{source_partition}_{system_id}_{config.example_class}.json").write_text(
        json.dumps(subset_summary, indent=2),
        encoding="utf-8",
    )

    print(f"Completed {system_id}. CSV path: {csv_path}")

print("TCAV completed for all requested systems.")
