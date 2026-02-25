from enum import Enum

FREQUENCY_BIN_COUNT = 72  # 72 for redimnet
SAMPLE_RATE = 16000

N_FFT = 512
WINDOW_LENGTH = 400
HOP_LENGTH = 240
MAX_SPECTOGRAM_DURATION_IN_SECONDS = (
    4.5  # Set to 7.06 for double_sentence training, for 4.5 otherwise.
)
F_MIN = 20
F_MAX = 7600

# Calculate the max number of samples for the target duration
MAX_SAMPLES = int(MAX_SPECTOGRAM_DURATION_IN_SECONDS * SAMPLE_RATE)

# Calculate the target number of frames for the spectrogram
# NOTE: For now, needs to be divisible by 8.
raw_frames = (
    MAX_SAMPLES - WINDOW_LENGTH
) // HOP_LENGTH + 1  # noam: this is the right formula for the number of frames
TOP_DB = 20

TARGET_FRAMES_CROP = (raw_frames // 8) * 8
TARGET_FRAMES_PAD = ((raw_frames + 7) // 8) * 8
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
