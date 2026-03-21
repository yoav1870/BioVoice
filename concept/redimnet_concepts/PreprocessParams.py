from enum import Enum

from pathlib import Path

# =========================
# AUDIO / MEL CONFIG
# =========================
FREQUENCY_BIN_COUNT = 72
SAMPLE_RATE = 16000

N_FFT = 512
WINDOW_LENGTH = 400  # 25 ms
HOP_LENGTH = 240  # 15 ms

F_MIN = 20
F_MAX = 7600

TOP_DB = 20

# =========================
# TARGET DURATION
# =========================
MAX_SPECTOGRAM_DURATION_IN_SECONDS = 4.5

# Convert duration → samples
MAX_SAMPLES = int(MAX_SPECTOGRAM_DURATION_IN_SECONDS * SAMPLE_RATE)

# Raw frame calculation (correct formula)
RAW_TARGET_FRAMES = (MAX_SAMPLES - WINDOW_LENGTH) // HOP_LENGTH + 1

# =========================
# 🔥 IMPORTANT (ReDimNet fix)
# Ensure divisible by 8
# =========================
TARGET_FRAMES = (RAW_TARGET_FRAMES // 8) * 8  # → 296

# =========================
# Optional: recompute exact duration
# =========================
ACTUAL_DURATION_SEC = TARGET_FRAMES * HOP_LENGTH / SAMPLE_RATE

print(f"Raw frames: {RAW_TARGET_FRAMES}")  # 299
print(f"Target frames: {TARGET_FRAMES}")  # 296
print(f"Actual duration: {ACTUAL_DURATION_SEC:.3f} sec")  # ~4.44


class LABEL_STRINGS(Enum):
    ANGRY = "angry"
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CALM = "calm"
