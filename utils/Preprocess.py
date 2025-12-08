import logging
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import torch.nn.functional

from utils.PreprocessParams import *

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (optional, default is WARNING)
logger.setLevel(logging.DEBUG)

# Create a console handler (you can add more handlers for file or other destinations)
console_handler = logging.StreamHandler()

# Set the log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

def min_max_normalization(spectrogram: np.ndarray) -> np.ndarray:
    """
    Scales the spectrogram values between 0 and 1.
    """
    return (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

def standardization(spectrogram: np.ndarray) -> np.ndarray:
    """
    Standardizes the spectrogram to have zero mean and unit variance.
    """
    return (spectrogram - spectrogram.mean()) / spectrogram.std()

def scale_between_minus_one_and_one(spectrogram: np.ndarray) -> np.ndarray:
    """
    Scales the spectrogram values between -1 and 1.
    """
    return 2 * (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min()) - 1


def audio_to_waveform(file_path: Path, target_sample_rate: int = SAMPLE_RATE):
    """
    Given audio file path and target sample rate, extracts waveform data.
    :param file_path: Audio file path.
    :param target_sample_rate: Desired sample rate.
    :return: Waveform data and the target sample rate.
    """
    waveform, loaded_sample_rate = librosa.load(file_path, mono=True, sr=target_sample_rate)
    logger.debug(f"Loaded audio file with shape: {waveform.shape}")

    return waveform, loaded_sample_rate

def trim_silence(waveform: np.ndarray, top_db: int = 20) -> np.ndarray:
    trimmed_waveform, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed_waveform


def audio_to_mel_spectrogram(file_path: Path,
                            sample_rate: int = SAMPLE_RATE,
                            n_fft = N_FFT,
                            window_length = WINDOW_LENGTH,
                            hop_length = HOP_LENGTH,
                            n_mels: int = FREQUENCY_BIN_COUNT,
                            max_length_in_seconds: float = MAX_SPECTOGRAM_DURATION_IN_SECONDS,
                            resizing = True,
                            top_db: int = TOP_DB,
                            normalization_fn: Callable[[np.ndarray], np.ndarray] = lambda x: x):
    """
    Given audio file path, extracts its waveform and from it creates a mel-spectrogram.
    :param resizing: True if you wish to resize the spectrogram to a fixed length.
    :param file_path: Audio file path.
    :param sample_rate: Desired sample rate.
    :param n_fft: Number of fft values. Defaults to the N_FFT preprocess macro.
    :param window_length: Spectrogram window length. Defaults to WINDOW_LENGTH marco.
    :param hop_length: Hop length in frames for spectrogram.
    :param n_mels: Number of frequency bins for the spectrogram. Defaults to FREQUENCY_BIN_COUNT macro.
    :param max_length_in_seconds: Limit on the length of spectrogram in seconds. Defaults to MAX_SPECTROGRAM_DURATION_IN_SECONDS
    :param normalization_fn: Function that normalizes the spectrogram. Defaults to standardization.
    :return: Mel-spectrogram with the desired attributes.
    """
    waveform, sample_rate = audio_to_waveform(file_path, sample_rate)

    #waveform = trim_silence(waveform, top_db=top_db)
    #waveform = librosa.effects.preemphasis(waveform)

    mel_spectrogram = librosa.feature.melspectrogram(y=waveform,
                                                     sr=sample_rate,
                                                     n_mels=n_mels,
                                                     n_fft=n_fft,
                                                     win_length = window_length,
                                                     hop_length = hop_length,
                                                     power=2.0,
                                                     center=False,
                                                     )

    if resizing:
        mel_spectrogram = resize_spectrogram_to_max_duration(spectrogram=mel_spectrogram,
                                                                 max_duration_seconds=max_length_in_seconds,
                                                                 sample_rate=sample_rate,
                                                                 win_length=window_length,
                                                                 hop_length=hop_length)

    logger.debug(f"Spectrogram shape at return: {mel_spectrogram.shape}")

    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    mel_spectrogram = normalization_fn(mel_spectrogram)

    return mel_spectrogram


def resize_spectrogram_to_max_duration(spectrogram, max_duration_seconds, sample_rate, win_length, hop_length):
    """
    Pads or truncates a spectrogram to match the maximum duration in seconds.

    Args:
        spectrogram (torch.Tensor): Input spectrogram (shape: [channels, n_mels, n_frames]).
        max_duration_seconds (float): Target maximum duration in seconds.
        sample_rate (int): Sampling rate of the audio signal.
        win_length (int): Window size used in the STFT.
        hop_length (int): Hop size used in the STFT.

    Returns:
        torch.Tensor: Padded spectrogram with the frame count to achieve the asked duration.
    """

    # compute the target number of frames
    max_samples = int(max_duration_seconds * sample_rate)
    target_frames = (max_samples - win_length) // hop_length + 1

    if spectrogram.shape[1] > target_frames: # Truncate
        logger.debug("Applied truncation")
        spectrogram = spectrogram[:, :target_frames]
    else: # Pad spectrogram.shape[1] <= target_frames
        logger.debug("Applied padding")
        padding = target_frames - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')

    return spectrogram