from enum import Enum


# =========================
# AUDIO / MEL CONFIG
# =========================
FREQUENCY_BIN_COUNT = 80
SAMPLE_RATE = 16000

N_FFT = 512
WINDOW_LENGTH = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms

F_MIN = 20
F_MAX = 7600

TOP_DB = 20

# =========================
# TARGET DURATION
# =========================
MAX_SPECTOGRAM_DURATION_IN_SECONDS = 4.5
MAX_SAMPLES = int(MAX_SPECTOGRAM_DURATION_IN_SECONDS * SAMPLE_RATE)

# Reference: the simple non-centered formula yields 299 frames.
RAW_TARGET_FRAMES = (MAX_SAMPLES - WINDOW_LENGTH) // HOP_LENGTH + 1

# Verified from the official ReDimNet frontend (`model.spec`) with center=True.
# For a 4.5-second waveform (72000 samples), the mel output is (1, 72, 301).
TARGET_FRAMES = 200


class LABEL_STRINGS(Enum):
    ANGRY = "angry"
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CALM = "calm"
