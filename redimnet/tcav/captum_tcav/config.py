from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

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
