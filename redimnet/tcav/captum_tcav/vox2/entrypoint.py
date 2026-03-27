# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false

from __future__ import annotations

import os
import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
TEMP_DIR = RUNTIME_DIR / "tmp"
CAV_SAVE_PATH = RUNTIME_DIR / "cav_cache"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
CAV_SAVE_PATH.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)

from captum.concept import Concept, TCAV

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from captum_tcav.common import chunk_paths, predict_argmax_target, resolve_layers
    from captum_tcav.concepts import make_iter
    from captum_tcav.export_csv import (
        CsvRow,
        aggregate_rows_by_weight,
        flatten_scores_rows,
        write_scores_csv,
    )
    from captum_tcav.vox2.config import load_config
    from captum_tcav.vox2.data import discover_speakers, load_input, validate_config
    from captum_tcav.vox2.modeling import load_model
else:
    from ..common import chunk_paths, predict_argmax_target, resolve_layers
    from ..concepts import make_iter
    from ..export_csv import (
        CsvRow,
        aggregate_rows_by_weight,
        flatten_scores_rows,
        write_scores_csv,
    )
    from .config import load_config
    from .data import discover_speakers, load_input, validate_config
    from .modeling import load_model


config = load_config()
validate_config(config)

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

model = load_model(config)
resolved_layers = resolve_layers(model, config.layers)
speakers = discover_speakers(config.data_dir)
mytcav = TCAV(
    model=model,
    layers=resolved_layers,
    save_path=str(CAV_SAVE_PATH),
)
all_rows: list[CsvRow] = []
for index, (speaker_name, speaker_paths) in enumerate(speakers.items(), start=1):
    speaker_paths = speaker_paths[: config.max_clips_per_speaker]
    speaker_chunks = chunk_paths(speaker_paths, config.max_clips_per_chunk)
    print(
        f"Processing speaker {index}/{len(speakers)}: {speaker_name} ({len(speaker_paths)} clips)"
    )
    inputs = load_input(config, model, speaker_chunks[0])
    pred_target_idx = predict_argmax_target(model, inputs)
    speaker_rows: list[tuple[list[CsvRow], int]] = []
    for chunk_paths_for_speaker in speaker_chunks:
        inputs = load_input(config, model, chunk_paths_for_speaker)
        scores = mytcav.interpret(inputs, experimental_sets, target=pred_target_idx)
        speaker_rows.append(
            (
                flatten_scores_rows(
                    scores=scores,
                    experimental_sets=experimental_sets,
                    speaker=speaker_name,
                    pred_target_idx=pred_target_idx,
                ),
                len(chunk_paths_for_speaker),
            )
        )
    all_rows.extend(aggregate_rows_by_weight(speaker_rows))
csv_path = write_scores_csv(
    rows=all_rows,
    output_path=Path(__file__).resolve().parent / "output" / "tcav_scores.csv",
)

print("TCAV completed.")
print(f"CSV path: {csv_path}")
