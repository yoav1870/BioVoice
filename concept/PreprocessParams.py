from enum import Enum

FREQUENCY_BIN_COUNT = 72
SAMPLE_RATE = 16000

N_FFT = 512
WINDOW_LENGTH = N_FFT
HOP_LENGTH = N_FFT // 2
MAX_SPECTOGRAM_DURATION_IN_SECONDS = (
    4.5  # Set to 7.06 for double_sentence training, for 4.5 otherwise.
)

# Calculate the max number of samples for the target duration
MAX_SAMPLES = int(MAX_SPECTOGRAM_DURATION_IN_SECONDS * SAMPLE_RATE)

# Calculate the target number of frames for the spectrogram
# NOTE: For now, needs to be divisible by 8.
TARGET_FRAMES = (
    MAX_SAMPLES - WINDOW_LENGTH
) // HOP_LENGTH + 1  # noam: this is the right formula for the number of frames
TOP_DB = 20

# print(TARGET_FRAMES) # debug


class LABEL_STRINGS(Enum):
    ANGRY = "angry"
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CALM = "calm"
