from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

concept_root = Path("/home/SpeakerRec/BioVoice/concept/resnet_293_concepts/concepts")
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

# WeSpeaker ResNet293 model paths
wespeaker_model_dir = Path(
    "/home/SpeakerRec/BioVoice/data/models/wespeaker_resnet293_lm"
)
wespeaker_models_dir = Path(
    "/home/SpeakerRec/BioVoice/external/wespeaker/wespeaker/models"
)

layers = ["layer4"]
random_seed = 1337

plan_base = Path(
    "/home/SpeakerRec/BioVoice/data/datasets/ASVspoof5_tars/ASVspoof5_protocols/train_dev_16_systems_outputs"
)
trained_models_base = Path(
    "/home/SpeakerRec/BioVoice/data/models/asvspoof5_train_dev_16_systems"
)
global_model_dir = Path(
    "/home/SpeakerRec/BioVoice/data/models/asvspoof5_train_dev_16_systems_global_excluding_a12_resnet293"
)
model_loading_mode = "global"  # options: "per_system", "global"
system_ids = [
    f"A{i:02d}" for i in range(1, 17)  # if i != 12
]  # Exclude A12 since it has very little data
split_name = "test"  # options: "train", "test"
example_class = (
    "spoof"  # options when target_class_mode == "single": "spoof", "bonafide"
)
target_class_mode = "both"  # options: "single", "both"
output_mode = "row"  # options: "mean", "row"
subset_seed = 42
subset_num_speakers = 40
subset_utts_per_speaker = 20
subset_min_utts_per_speaker = 20
max_clips_per_chunk = 20
save_predictions = False
fixed_train_speakers: list[str] = []
fixed_dev_speakers: list[str] = []
excluded_train_speakers: list[str] = []
excluded_dev_speakers: list[str] = []
output_subdir = "new_subset_20spk_20utts_per_spk_one_logistic_head"

# fbank feature extraction parameters (must match model training)
fbank_num_mel_bins = 80
fbank_frame_length = 25
fbank_frame_shift = 10
fbank_sample_frequency = 16000


@dataclass(frozen=True)
class Config:
    concept_root: Path
    concept_names: list[str]
    random_concept_name: str
    wespeaker_model_dir: Path
    wespeaker_models_dir: Path
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
    fbank_num_mel_bins: int
    fbank_frame_length: int
    fbank_frame_shift: int
    fbank_sample_frequency: int
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
        wespeaker_model_dir=Path(wespeaker_model_dir),
        wespeaker_models_dir=Path(wespeaker_models_dir),
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
        fbank_num_mel_bins=int(fbank_num_mel_bins),
        fbank_frame_length=int(fbank_frame_length),
        fbank_frame_shift=int(fbank_frame_shift),
        fbank_sample_frequency=int(fbank_sample_frequency),
        max_clips_per_chunk=int(max_clips_per_chunk),
        save_predictions=bool(save_predictions),
        fixed_train_speakers=list(fixed_train_speakers),
        fixed_dev_speakers=list(fixed_dev_speakers),
        excluded_train_speakers=list(excluded_train_speakers),
        excluded_dev_speakers=list(excluded_dev_speakers),
    )
