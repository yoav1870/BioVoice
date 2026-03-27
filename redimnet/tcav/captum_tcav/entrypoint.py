# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false

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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from captum_tcav.data_hooks import (
        discover_speakers,
        chunk_paths,
        load_config,
        load_input,
        load_model,
        make_iter,
        predict_target_index,
        resolve_layers,
        validate_config,
    )
    from captum_tcav.export_csv import (
        CsvRow,
        aggregate_rows_by_weight,
        flatten_scores_rows,
        write_scores_csv,
    )
else:
    from .data_hooks import (
        discover_speakers,
        chunk_paths,
        load_config,
        load_input,
        load_model,
        make_iter,
        predict_target_index,
        resolve_layers,
        validate_config,
    )
    from .export_csv import (
        CsvRow,
        aggregate_rows_by_weight,
        flatten_scores_rows,
        write_scores_csv,
    )


config = load_config()
validate_config(config)

concepts = [
    Concept(idx, concept_name, make_iter(concept_name, config))
    for idx, concept_name in enumerate(config.concept_names)
]
random_data_iter = make_iter(config.random_concept_name, config)
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
    pred_target_idx = predict_target_index(model, inputs)
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
