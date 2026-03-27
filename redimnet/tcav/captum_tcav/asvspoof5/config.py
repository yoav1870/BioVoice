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
trained_models_base = Path("/home/SpeakerRec/BioVoice/data/models/asvspoof5_train_dev_16_systems")
system_ids = [f"A{i:02d}" for i in range(1, 17)]
split_name = "test"
target_class = "spoof"
subset_seed = 42
subset_num_speakers = 1
subset_utts_per_speaker = 4
subset_min_utts_per_speaker = 4
max_clips_per_chunk = 8
save_predictions = False


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
    system_ids: list[str]
    split_name: str
    target_class: str
    subset_seed: int
    subset_num_speakers: int
    subset_utts_per_speaker: int
    subset_min_utts_per_speaker: int
    max_clips_per_chunk: int = 8
    save_predictions: bool = False


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
        system_ids=list(system_ids),
        split_name=str(split_name),
        target_class=str(target_class),
        subset_seed=int(subset_seed),
        subset_num_speakers=int(subset_num_speakers),
        subset_utts_per_speaker=int(subset_utts_per_speaker),
        subset_min_utts_per_speaker=int(subset_min_utts_per_speaker),
        max_clips_per_chunk=int(max_clips_per_chunk),
        save_predictions=bool(save_predictions),
    )
