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
data_dir = Path(
    "/home/SpeakerRec/BioVoice/data/datasets/voxceleb2/voxceleb2_hf/extracted_aac/aac"
)
head_path = Path(
    "/home/SpeakerRec/BioVoice/data/heads/redim_speaker_head_vox2_100_50.pt"
)
target_index = 0
target_mode = "predicted"
model_repo = "IDRnD/ReDimNet"
model_entrypoint = "ReDimNet"
model_name = "b6"
model_train_type = "ptn"
model_dataset = "vox2"
layers = ["stage4"]
random_seed = 1337
max_clips_per_speaker = 50
max_clips_per_chunk = 8


@dataclass(frozen=True)
class Config:
    concept_root: Path
    concept_names: list[str]
    random_concept_name: str
    data_dir: Path
    head_path: Path
    target_index: int
    target_mode: str
    model_repo: str
    model_entrypoint: str
    model_name: str
    model_train_type: str
    model_dataset: str
    layers: list[str]
    random_seed: int
    max_clips_per_speaker: int = 32
    max_clips_per_chunk: int = 8


def load_config() -> Config:
    return Config(
        concept_root=Path(concept_root),
        concept_names=list(concept_names),
        random_concept_name=random_concept_name,
        data_dir=Path(data_dir),
        head_path=Path(head_path),
        target_index=int(target_index),
        target_mode=str(target_mode),
        model_repo=model_repo,
        model_entrypoint=model_entrypoint,
        model_name=model_name,
        model_train_type=model_train_type,
        model_dataset=model_dataset,
        layers=list(layers),
        random_seed=int(random_seed),
        max_clips_per_speaker=int(max_clips_per_speaker),
        max_clips_per_chunk=int(max_clips_per_chunk),
    )
