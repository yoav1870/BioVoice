"""GradCAM analysis configuration for ResNet293 on ASVspoof5."""

from pathlib import Path

# ===== Paths =====
BIOVOICE_ROOT = Path("/home/SpeakerRec/BioVoice")
TCAV_DIR = BIOVOICE_ROOT / "resnet_293" / "tcav"
OUTPUT_DIR = BIOVOICE_ROOT / "resnet_293" / "gradcam" / "output"

# ===== Analysis settings =====
TARGET_SYSTEM = "A07"
GRADCAM_LAYER = "layer4"
THRESHOLD = 0.70

# How many speakers / utterances to process (keep small since CPU-only)
MAX_SPEAKERS_PER_CLASS = 5
MAX_UTTS_PER_SPEAKER = 3
