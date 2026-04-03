from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

concept_root = Path("/home/SpeakerRec/BioVoice/concept/redimnet_concepts/concepts")
concept_names = [
    "long_constant_thick",
    "long_dropping_flat_thick",
    "long_dropping_steep_thick",
    "long_dropping_steep_thin",
    "long_rising_flat_thick",
    "long_rising_steep_thick",
    "long_rising_steep_thin",
    "short_constant_thick",
    "short_dropping_steep_thick",
    "short_dropping_steep_thin",
    "short_rising_steep_thick",
    "short_rising_steep_thin",
]
random_concept_name = "random_0"
model_repo = "IDRnD/ReDimNet"
model_entrypoint = "ReDimNet"
model_name = "b6"
model_train_type = "ptn"
model_dataset = "vox2"
layers = ["stage4"]
random_seed = 1337

plan_base = Path(
    "/home/SpeakerRec/BioVoice/data/datasets/ASVspoof5_tars/ASVspoof5_protocols/train_dev_16_systems_outputs"
)
trained_models_base = Path(
    "/home/SpeakerRec/BioVoice/data/models/asvspoof5_train_dev_16_systems"
)
global_model_dir = Path(
    "/home/SpeakerRec/BioVoice/data/models/asvspoof5_train_dev_16_systems_global_excluding_a12"
)
model_loading_mode = "global"  # options: "per_system", "global"
system_ids = [f"A{i:02d}" for i in range(1, 17)]
split_name = "test"  # options: "train", "test"
example_class = (
    "spoof"  # options when target_class_mode == "single": "spoof", "bonafide"
)
target_class_mode = "both"  # options: "single", "both"
output_mode = "row"  # options: "mean", "row"
subset_seed = 42
subset_num_speakers = 20
subset_utts_per_speaker = 20
subset_min_utts_per_speaker = 20
max_clips_per_chunk = 10
save_predictions = False
fixed_train_speakers: list[str] = []
fixed_dev_speakers: list[str] = []
excluded_train_speakers: list[str] = []
excluded_dev_speakers: list[str] = []
# excluded_train_speakers = [
#     "T_0380",
#     "T_0411",
#     "T_0635",
#     "T_0897",
#     "T_1864",
#     "T_2149",
#     "T_2284",
#     "T_2326",
#     "T_2791",
#     "T_3455",
#     "T_3714",
#     "T_3850",
#     "T_3883",
#     "T_4049",
#     "T_4126",
#     "T_4175",
#     "T_4618",
#     "T_4769",
#     "T_4913",
#     "T_5053",
# ]
# excluded_dev_speakers = [
#     "D_0430",
#     "D_0461",
#     "D_0546",
#     "D_0956",
#     "D_2288",
#     "D_2884",
#     "D_2937",
#     "D_2975",
#     "D_3192",
#     "D_3501",
#     "D_3668",
#     "D_3927",
#     "D_3964",
#     "D_4023",
#     "D_4057",
#     "D_4814",
#     "D_4825",
#     "D_4888",
#     "D_5112",
#     "D_5248",
# ]
output_subdir = "subset_20spk_20utts_per_spk_one_logistic_head"

# Useful combinations:
# - Per-system legacy averages:
#   model_loading_mode = "per_system"
#   target_class_mode = "single"
#   example_class = "spoof"  # or "bonafide"
#   output_mode = "mean"
#
# - Global shared model, utterance-level export for both classes:
#   model_loading_mode = "global"
#   target_class_mode = "both"
#   split_name = "test"
#   output_mode = "row"


@dataclass(frozen=True)
class Config:
    concept_root: Path
    concept_names: list[str]
    random_concept_name: str
    model_repo: str
    model_entrypoint: str
    model_name: str
    model_train_type: str
    model_dataset: str
    layers: list[str]
    random_seed: int
    plan_base: Path
    trained_models_base: Path
    global_model_dir: Path
    model_loading_mode: str
    system_ids: list[str]
    split_name: str
    example_class: str
    target_class_mode: str
    output_mode: str
    subset_seed: int
    subset_num_speakers: int
    subset_utts_per_speaker: int
    subset_min_utts_per_speaker: int
    output_subdir: str
    max_clips_per_chunk: int = 8
    save_predictions: bool = False
    fixed_train_speakers: list[str] | None = None
    fixed_dev_speakers: list[str] | None = None
    excluded_train_speakers: list[str] | None = None
    excluded_dev_speakers: list[str] | None = None


def load_config() -> Config:
    return Config(
        concept_root=Path(concept_root),
        concept_names=list(concept_names),
        random_concept_name=random_concept_name,
        model_repo=model_repo,
        model_entrypoint=model_entrypoint,
        model_name=model_name,
        model_train_type=model_train_type,
        model_dataset=model_dataset,
        layers=list(layers),
        random_seed=int(random_seed),
        plan_base=Path(plan_base),
        trained_models_base=Path(trained_models_base),
        global_model_dir=Path(global_model_dir),
        model_loading_mode=str(model_loading_mode),
        system_ids=list(system_ids),
        split_name=str(split_name),
        example_class=str(example_class),
        target_class_mode=str(target_class_mode),
        output_mode=str(output_mode),
        subset_seed=int(subset_seed),
        subset_num_speakers=int(subset_num_speakers),
        subset_utts_per_speaker=int(subset_utts_per_speaker),
        subset_min_utts_per_speaker=int(subset_min_utts_per_speaker),
        output_subdir=str(output_subdir),
        max_clips_per_chunk=int(max_clips_per_chunk),
        save_predictions=bool(save_predictions),
        fixed_train_speakers=list(fixed_train_speakers),
        fixed_dev_speakers=list(fixed_dev_speakers),
        excluded_train_speakers=list(excluded_train_speakers),
        excluded_dev_speakers=list(excluded_dev_speakers),
    )
