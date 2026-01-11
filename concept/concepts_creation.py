import math
from pathlib import Path
from re import L, S
from tkinter import N
from typing import List, Optional, Tuple
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp, square
from sympy import sec
from torch import long
from Preprocess import audio_to_mel_spectrogram
from Visualizations import plot_mel_spectrogram

from PreprocessParams import (
    MAX_SPECTOGRAM_DURATION_IN_SECONDS,
    TARGET_FRAMES_PAD,
    FREQUENCY_BIN_COUNT,
    HOP_LENGTH,
    SAMPLE_RATE,
    F_MIN,
    F_MAX,
)


# --- Your concept knobs (means + stds) ---
SHORT_TIME = 0.05  # seconds
LONG_TIME = 0.1  # seconds
STD_DEV_TIME = 0.008  # seconds (clamped in code)

STEEP_RISING_DEGREE_TONE = 32  # deg
FLAT_RISING_DEGREE_TONE = 10  # deg
STD_DEV_RISING = 7  # deg

STEEP_DROPPING_DEGREE_TONE = -80  # deg
FLAST_DROPPING_DEGREE_TONE = -20  # deg
STD_DEV_DROPPING = 15  # deg

CONSTANT_DEGREE_TONE = 0  # deg
STD_DEV_CONSTANT = 15  # deg

THIN_THICKNESS = 0.2  # ~pixels (FWHM)
THICK_THICKNESS = 1  # ~pixels (FWHM)
RANDOM_THICKNESS_BOUNDARY = 1  # pixels (FWHM)
MIN_FWHM = 0.05
# ----------------------------
# Internals
# ----------------------------

TARGET_FRAMES = TARGET_FRAMES_PAD
FRAMES = TARGET_FRAMES
FREQS = FREQUENCY_BIN_COUNT
DT = HOP_LENGTH / SAMPLE_RATE  # seconds per frame
FPS = 1.0 / DT  # frames per second


def _clip(v, lo, hi):
    """Clamp a value *v* to the interval [lo, hi]."""
    return max(lo, min(hi, v))


def _trunc_normal(
    rng: np.random.Generator, mean: float, std: float, lo: float, hi: float
) -> float:
    """Sample from a normal distribution N(mean, std) but truncate/clamp the result into [lo, hi]."""
    x = rng.normal(mean, std)
    return _clip(x, lo, hi)


def _draw_line_minwidth(
    S: np.ndarray,
    t0: int,
    L: int,
    y0: float,
    slope: float,
    *,
    fwhm_px: float,
    amp: float = 1.0,
    vibrato_amp: float = 0.0,
    vibrato_freq: float = 0.0,
):
    # FWHM -> sigma
    sigma = max(1e-3, float(fwhm_px) / 2.355)

    # how many neighbor bins to touch
    K = max(1, int(np.ceil(3.0 * sigma)))  # 3*sigma support

    t_idx = np.arange(L, dtype=np.int32)
    cols = t0 + t_idx
    valid = (cols >= 0) & (cols < FRAMES)
    cols = cols[valid]
    t_idx = t_idx[valid]
    if cols.size == 0:
        return

    y = y0 + slope * t_idx.astype(np.float32)
    if vibrato_amp > 0 and vibrato_freq > 0:
        t_sec = t_idx * DT
        y += vibrato_amp * np.sin(2 * np.pi * vibrato_freq * t_sec)
    y_base = np.floor(y).astype(np.int32)
    w = (y - y_base).astype(np.float32)

    # keep neighbors valid
    y_base = np.clip(y_base, K, FREQS - 2 - K)

    offs = np.arange(-K, K + 1, dtype=np.int32)
    g = np.exp(-0.5 * (offs.astype(np.float32) / sigma) ** 2).astype(np.float32)
    g /= g.sum() + 1e-12  # normalize so thin/thick differ in width, not total energy

    for oi, off in enumerate(offs):
        S[y_base + off, cols] += amp * (1.0 - w) * g[oi]
        S[y_base + 1 + off, cols] += amp * w * g[oi]


def _duration_frames_for_seconds(
    seconds: float, std_dev_time: float, rng: np.random.Generator
) -> int:
    """
    Convert seconds count into an integer number of frames.
    Uses truncated normal sampling with jitter and ensures the result is at least 2 frames
    and at most 25% of the spectrogram duration.
    """
    mean_s = seconds
    min_s = 2 * DT
    max_s = max(DT * 3, 0.25 * MAX_SPECTOGRAM_DURATION_IN_SECONDS)
    dur_s = _trunc_normal(rng, mean_s, std_dev_time, min_s, max_s)
    frames = int(round(dur_s * FPS))
    return int(_clip(frames, 2, FRAMES - 2))


# input duration label, returns int number of frames.
def _duration_frames_for_label(label: str, rng: np.random.Generator) -> int:
    """
    Convert a duration label ("short" or "long") into an integer number of frames.
    Uses truncated normal sampling with jitter and ensures the result is at least 2 frames
    and at most 25% of the spectrogram duration.
    """
    mean_s = SHORT_TIME if label == "short" else LONG_TIME
    return _duration_frames_for_seconds(mean_s, STD_DEV_TIME, rng)


# input: raw mean angle and it's std_dev, output: random angle.
def _angle_deg_for_raw_input(
    mean_tone_degree: float, std_tone_degree: float, rng: np.random.Generator
) -> float:
    """
    Sample an angle (in degrees) from a raw input value and its standard deviation.
    """
    return _trunc_normal(rng, mean_tone_degree, std_tone_degree, -89.0, 89.0)


# get tone and rate factors and returns the corresponding angle in degrees
def _angle_deg_for_factors(tone: str, rate: str, rng: np.random.Generator) -> float:
    """
    Sample an angle (in degrees) that represents the slope of a tone shift.
    - tone: "rising", "dropping", or "constant"
    - rate: "steep" or "flat" (ignored if tone == "constant")
    Returns a truncated normal sample in [-89, 89] degrees.
    """
    if tone == "constant":
        mean, std = CONSTANT_DEGREE_TONE, STD_DEV_CONSTANT
    elif tone == "rising":
        mean = STEEP_RISING_DEGREE_TONE if rate == "steep" else FLAT_RISING_DEGREE_TONE
        std = STD_DEV_RISING
    elif tone == "dropping":
        mean = (
            STEEP_DROPPING_DEGREE_TONE
            if rate == "steep"
            else FLAST_DROPPING_DEGREE_TONE
        )
        std = STD_DEV_DROPPING
    else:
        raise ValueError(f"Unknown tone '{tone}'")
    return _trunc_normal(rng, mean, std, -89.0, 89.0)


# converts thickness label to "full width at half maximum" (FWHM)
# e.g. "thin" to 2 pixels.
def _thickness_from_label(thick_label: str, rng: np.random.Generator) -> float:
    """
    Convert a thickness label ("thin" or "thick") into a Gaussian full width at half maximum (FWHM).
    Adds ±0.5px jitter and clamps to [1, 32] pixels.
    """
    base = THIN_THICKNESS if thick_label == "thin" else THICK_THICKNESS
    return float(_clip(base + rng.uniform(-0.5, 0.5), MIN_FWHM, 32.0))


# converts mean thickness to random thickness.
def _thickness_from_thickness(
    mean_thickness: float,
    rng: np.random.Generator,
    random_thickness_boundary: float = 0.5,
) -> float:
    return float(
        _clip(
            mean_thickness
            + rng.uniform(-random_thickness_boundary, random_thickness_boundary),
            MIN_FWHM,
            32.0,
        )
    )


# writes the directory name for a concept based on its factors
def _dir_name(
    length: str, tone: str, rate: str, thickness: str, isVibrato: bool
) -> str:
    """Build the canonical directory name string for a concept based on its factors."""
    # return (
    #     f"{length}_{tone}_{rate}_{thickness}"
    #     if tone != "constant"
    #     else f"{length}_{tone}_{thickness}"
    # )
    if not isVibrato:
        return (
            f"{length}_{tone}_{rate}_{thickness}"
            if tone != "constant"
            else f"{length}_{tone}_{thickness}"
        )
    else:
        return (
            f"{length}_{tone}_{rate}_{thickness}_Vibrato"
            if tone != "constant"
            else f"{length}_{tone}_{thickness}_Vibrato"
        )


# convert angle in degrees to slope in dy/dx
def _slope_pixels_per_frame(angle_deg: float) -> float:
    """
    Convert an angle in degrees into slope in pixel coordinates (dy/dx) in units of frequency bins per frame.
    """
    return math.tan(math.radians(angle_deg))


# IDK
def _valid_y0_range(slope: float, L: int) -> Tuple[float, float]:
    """
    Given a slope and line length L, compute the valid range of starting y0 values such that
    the line segment fits entirely within [0, FREQS-1].
    """
    if slope >= 0:
        y0_min = 0.0
        y0_max = (FREQS - 1) - slope * (L - 1)
    else:
        y0_min = -slope * (L - 1)
        y0_max = FREQS - 1
    return (max(0.0, y0_min), min(FREQS - 1.0, y0_max))


# deterministic function!
def _draw_line_gaussian(
    S: np.ndarray, t0: int, L: int, y0: float, slope: float, fwhm_px: float
):
    """
    Draw a Gaussian-blurred line segment into spectrogram array *S*.

    Args:
        S: 2D ndarray of shape [FREQS, FRAMES]. Will be modified in-place.
        t0: Starting column index (time frame).
        L:  Length of the segment in frames.
        y0: Starting row index (frequency bin, may be float).
        slope: Pixels per frame (dy/dx).
        fwhm_px: Desired full width at half max (Gaussian thickness in pixels).
    """
    sigma = max(0.3, fwhm_px / 2.355)
    t_idx = np.arange(L, dtype=np.float32)
    y_center = y0 + slope * t_idx
    cols = (t0 + t_idx).astype(int)
    y_bins = np.arange(FREQS, dtype=np.float32)[:, None]
    mu = y_center[None, :]
    dist2 = (y_bins - mu) ** 2
    gauss = np.exp(-0.5 * dist2 / (sigma * sigma))
    amp = 0.9 + 0.2 * np.hanning(L)
    gauss *= amp[None, :]
    valid = (cols >= 0) & (cols < FRAMES)
    cols_v = cols[valid]
    if cols_v.size == 0:
        return
    S[:, cols_v] += gauss[:, valid]


def postprocess_like_redimnet(S: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Make synthetic patch closer to ReDimNet input space:
    - log(energy)
    - mean-normalize over time per mel bin (CMN)
    """
    S = np.maximum(S, 0.0)
    S = np.log(S + eps)
    S = S - S.mean(axis=1, keepdims=True)
    return S.astype(np.float32)


def pad_or_crop_frames(S: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad (with 0.0) or crop on the time axis to target_frames."""
    f, t = S.shape
    if t == target_frames:
        return S
    if t > target_frames:
        return S[:, :target_frames]
    out = np.zeros((f, target_frames), dtype=S.dtype)  # 0.0 is good after CMN
    out[:, :t] = S
    return out


def generate_concept_patch_for_raw_input(
    mean_length: float,
    std_length: float,
    mean_tone_degree: float,
    std_tone_degree: float,
    mean_thickness: float,
    rng: Optional[np.random.Generator] = None,
    random_thickness_boundary: float = 0.5,
    vibrato_amp: float = 0.0,
    vibrato_freq: float = 0.0,
) -> np.ndarray:
    rng = rng or np.random.default_rng()

    # start with small noise in "energy space"
    S = np.zeros((FREQS, FRAMES), dtype=np.float32)

    L = _duration_frames_for_seconds(mean_length, std_length, rng)
    ang = _angle_deg_for_raw_input(mean_tone_degree, std_tone_degree, rng)
    k = _slope_pixels_per_frame(ang)
    fwhm = _thickness_from_thickness(mean_thickness, rng, random_thickness_boundary)

    t0 = rng.integers(0, FRAMES - L)
    ylo, yhi = _valid_y0_range(k, L)
    y0 = (FREQS - 1) / 2.0 if (yhi < ylo + 1e-5) else rng.uniform(ylo, yhi)

    _draw_line_minwidth(
        S,
        int(t0),
        int(L),
        float(y0),
        float(k),
        fwhm_px=fwhm,
        vibrato_amp=vibrato_amp,
        vibrato_freq=vibrato_freq,
    )

    # IMPORTANT: remove min-max normalization
    # Instead convert to ReDim-like space:
    S = postprocess_like_redimnet(S)

    # (Optional) ensure exact framing if you later switch between crop/pad
    S = pad_or_crop_frames(S, FRAMES)

    return S


def generate_concept_patch_for_labels(
    length: str,
    tone: str,
    rate: str,
    thickness: str,
    rng: Optional[np.random.Generator] = None,
    vibrato_amp: float = 0.0,
    vibrato_freq: float = 0.0,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    S = rng.normal(0.0, 0.02, size=(FREQS, FRAMES)).astype(np.float32)

    L = _duration_frames_for_label(length, rng)
    ang = _angle_deg_for_factors(tone, rate, rng)
    k = _slope_pixels_per_frame(ang)
    fwhm = _thickness_from_label(thickness, rng)

    t0 = rng.integers(0, FRAMES - L)
    ylo, yhi = _valid_y0_range(k, L)
    y0 = (FREQS - 1) / 2.0 if (yhi < ylo + 1e-5) else rng.uniform(ylo, yhi)

    _draw_line_minwidth(
        S,
        int(t0),
        int(L),
        float(y0),
        float(k),
        fwhm_px=fwhm,
        vibrato_amp=0.0,
        vibrato_freq=0.0,
    )

    S = postprocess_like_redimnet(S)
    S = pad_or_crop_frames(S, FRAMES)
    return S


def create_concept_dir(
    base_dir: Path,
    length: str,
    tone: str,
    rate: str,
    thickness: str,
    mean_length: float,
    std_length: float,
    mean_tone_degree: float,
    std_tone_degree: float,
    mean_thickness: float,
    n_samples: int = 100,
    seed: Optional[int] = None,
    random_thickness_boundary: float = 0.5,
    vibrato_amp: float = 0.0,
    vibrato_freq: float = 0.0,
    isVibrato: bool = False,
) -> Path:
    rng = np.random.default_rng(seed)
    concept_name = _dir_name(length, tone, rate, thickness, isVibrato)
    out_dir = base_dir / concept_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, n_samples + 1):
        S = generate_concept_patch_for_raw_input(
            mean_length=mean_length,
            std_length=std_length,
            mean_tone_degree=mean_tone_degree,
            std_tone_degree=std_tone_degree,
            mean_thickness=mean_thickness,
            rng=rng,
            random_thickness_boundary=random_thickness_boundary,  # FIX: was ignored before
            vibrato_amp=vibrato_amp,
            vibrato_freq=vibrato_freq,
        )
        np.save(out_dir / f"{i:06d}.npy", S)

    return out_dir


def show_arrays_in_separate_windows(
    arrays: List[np.ndarray], titles: List[str] = [], cmap: str = "magma"
):
    """
    Display each 2D ndarray in its own matplotlib window, all visible simultaneously.

    Args:
        arrays: List of 2D numpy arrays to display.
        titles: Optional list of window titles (same length as arrays).
        cmap: Colormap for visualization (default: 'magma').
    """
    if titles == []:
        titles = [f"Array {i+1}" for i in range(len(arrays))]

    for arr, title in zip(arrays, titles):
        plt.figure()  # create a new independent window
        plt.imshow(arr, aspect="auto", origin="lower", cmap=cmap)
        plt.colorbar()
        plt.title(title)

    # Show all figures at the same time
    plt.show(block=True)


def TEST_draw_line_gaussian():
    S = np.zeros((FREQUENCY_BIN_COUNT, TARGET_FRAMES), dtype=np.float32)
    _draw_line_gaussian(
        S,
        t0=0,
        L=30,
        y0=FREQUENCY_BIN_COUNT // 2,
        slope=_slope_pixels_per_frame(40),
        fwhm_px=3,
    )
    show_arrays_in_separate_windows([S], titles=["TEST_draw_line_gaussian"])


def TEST_generate_concept_patch():
    Lines_list = [
        generate_concept_patch_for_raw_input(
            # length="long",
            # tone="constant",
            # rate="ignored",
            # thickness="thick",
            mean_length=0.17,
            std_length=0.027,
            mean_tone_degree=0,
            std_tone_degree=1.3,
            mean_thickness=2.92,
            random_thickness_boundary=RANDOM_THICKNESS_BOUNDARY,
        )
        for _ in range(3)
    ]

    # # append other lines.
    # Lines_list.extend(
    #     generate_concept_patch_for_raw_input(
    #         # length="long",
    #         # tone="dropping",
    #         # rate="steep",
    #         # thickness="thick",
    #         mean_length=0.085,
    #         std_length=0.007,
    #         mean_tone_degree=-57,
    #         std_tone_degree=4,
    #         mean_thickness=4.9,
    #         random_thickness_boundary=0.2
    #     )
    #     for _ in range(3)
    # )

    # # plot Lines:
    # show_arrays_in_separate_windows(Lines_list)

    for i, line in enumerate(Lines_list):
        plot_mel_spectrogram(line, block=False)

    plot_mel_spectrogram(
        audio_to_mel_spectrogram(
            Path(r"RAVDESS\original_data\Actor_04\03-01-01-01-02-01-04.wav")
        ),
        block=False,
    )
    plot_mel_spectrogram(
        audio_to_mel_spectrogram(
            Path(r"RAVDESS\original_data\Actor_08\03-01-02-01-02-01-08.wav")
        ),
        block=False,
    )
    # plot_mel_spectrogram(audio_to_mel_spectrogram(Path(r"RAVDESS\original_data\Actor_16\03-01-02-01-01-01-16.wav")), block=False)
    plot_mel_spectrogram(
        audio_to_mel_spectrogram(
            Path(r"RAVDESS\original_data\Actor_15\03-01-02-01-02-01-15.wav")
        ),
        block=False,
    )

    plt.show()


def CREATE_ALL_CONCEPT_DIRS(out_dir: Path):
    random_seed = 42
    samples_count = 60

    ######## RISING: ########

    ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="long",
        tone="rising",
        rate="steep",
        thickness="thin",
        mean_length=0.18,
        std_length=0.01,
        mean_tone_degree=57,
        std_tone_degree=3.0,
        mean_thickness=1.5,
        random_thickness_boundary=0.2,
        n_samples=samples_count,
        seed=random_seed,
    )

    ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="long",
        tone="rising",
        rate="steep",
        thickness="thick",
        mean_length=0.18,
        std_length=0.02,
        mean_tone_degree=57,
        std_tone_degree=4,
        mean_thickness=3.0,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
    )

    # # ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="long",
        tone="rising",
        rate="flat",
        thickness="thick",
        mean_length=0.18,
        std_length=0.08,
        mean_tone_degree=33,
        std_tone_degree=8,
        mean_thickness=1.5,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
    )

    ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="short",
        tone="rising",
        rate="steep",
        thickness="thin",
        mean_length=0.09,
        std_length=0.01,
        mean_tone_degree=57,
        std_tone_degree=3.0,
        mean_thickness=1.5,
        random_thickness_boundary=0.2,
        n_samples=samples_count,
        seed=random_seed,
    )

    # ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="short",
        tone="rising",
        rate="steep",
        thickness="thick",
        mean_length=0.09,
        std_length=0.08,
        mean_tone_degree=33,
        std_tone_degree=8,
        mean_thickness=1.5,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
    )

    ######## CONSTANT: ########

    ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="long",
        tone="constant",
        rate="ignored",
        thickness="thick",
        mean_length=0.4,
        std_length=0.08,
        mean_tone_degree=0,
        std_tone_degree=0,
        mean_thickness=0.5,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
    )
    ##### VV #####

    create_concept_dir(
        base_dir=out_dir,
        length="long",
        tone="constant",
        rate="ignored",
        thickness="thick",
        isVibrato=True,
        mean_length=0.4,
        std_length=0.08,
        mean_tone_degree=0,
        std_tone_degree=0,
        mean_thickness=0.5,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
        vibrato_amp=1.0,
        vibrato_freq=5.0,
    )

    ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="short",
        tone="constant",
        rate="ignored",
        thickness="thick",
        mean_length=0.2,
        std_length=0.08,
        mean_tone_degree=0,
        std_tone_degree=0,
        mean_thickness=0.5,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
    )

    ######## Dropping: ########

    ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="long",
        tone="dropping",
        rate="steep",
        thickness="thin",
        mean_length=0.18,
        std_length=0.01,
        mean_tone_degree=-57,
        std_tone_degree=3.0,
        mean_thickness=1.5,
        random_thickness_boundary=0.2,
        n_samples=samples_count,
        seed=random_seed,
    )

    #### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="long",
        tone="dropping",
        rate="steep",
        thickness="thick",
        mean_length=0.18,
        std_length=0.02,
        mean_tone_degree=-57,
        std_tone_degree=4,
        mean_thickness=3.0,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
    )

    ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="long",
        tone="dropping",
        rate="flat",
        thickness="thick",
        mean_length=0.18,
        std_length=0.08,
        mean_tone_degree=-33,
        std_tone_degree=8,
        mean_thickness=1.5,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
    )
    create_concept_dir(
        base_dir=out_dir,
        length="long",
        tone="dropping",
        rate="flat",
        thickness="thick",
        isVibrato=True,
        mean_length=0.18,
        std_length=0.08,
        mean_tone_degree=-33,
        std_tone_degree=8,
        mean_thickness=1.5,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
        vibrato_amp=1.0,
        vibrato_freq=5.0,
    )

    ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="short",
        tone="dropping",
        rate="steep",
        thickness="thin",
        mean_length=0.09,
        std_length=0.01,
        mean_tone_degree=-57,
        std_tone_degree=3.0,
        mean_thickness=1.5,
        random_thickness_boundary=0.2,
        n_samples=samples_count,
        seed=random_seed,
    )

    # ##### VV #####
    create_concept_dir(
        base_dir=out_dir,
        length="short",
        tone="dropping",
        rate="steep",
        thickness="thick",
        mean_length=0.09,
        std_length=0.08,
        mean_tone_degree=-33,
        std_tone_degree=8,
        mean_thickness=1.5,
        random_thickness_boundary=0.3,
        n_samples=samples_count,
        seed=random_seed,
    )


def TEST_generate_random_pattern_spectrogram(pattern_type=None):
    random_negatives = [
        generate_random_pattern_spectrogram(pattern_type=pattern_type) for _ in range(3)
    ]

    for i, line in enumerate(random_negatives):
        plot_mel_spectrogram(line, block=False)

    plt.show()


def generate_random_pattern_spectrogram(
    freq_count=FREQS,
    frames=FRAMES,
    pattern_type: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a random/unrealistic spectrogram pattern for testing.

    Args:
        freq_count: Number of frequency bins (rows).
        frames: Number of time frames (columns).
        pattern_type: Type of pattern to generate:
            'solid', 'white_noise', 'dotted', 'stripes', or None for random choice.
        rng: Optional NumPy Generator for reproducibility.

    Returns:
        2D ndarray of shape [freq_count, frames] with values normalized to [0, 1].
    """
    rng = rng or np.random.default_rng()

    if pattern_type is None:
        pattern_type = rng.choice(["solid", "white_noise", "dotted", "stripes"])

    S = np.zeros((freq_count, frames), dtype=np.float32)

    if pattern_type == "solid":
        # Fill with a single random value
        S.fill(rng.uniform(0.2, 0.8))

    elif pattern_type == "white_noise":
        # Random Gaussian noise
        S = rng.normal(0.5, 0.25, size=(freq_count, frames)).astype(np.float32)

    elif pattern_type == "dotted":
        # Sparse dots across the spectrogram
        dot_count = rng.integers(freq_count * frames // 50, freq_count * frames // 20)
        for _ in range(dot_count):
            x = rng.integers(0, frames)
            y = rng.integers(0, freq_count)
            S[y, x] = rng.uniform(0.6, 1.0)

    elif pattern_type == "stripes":
        # Horizontal or vertical stripes
        stripe_orientation = rng.choice(["horizontal", "vertical"])
        stripe_width = rng.integers(1, max(2, min(freq_count, frames) // 10))
        if stripe_orientation == "horizontal":
            for y in range(0, freq_count, stripe_width * 2):
                S[y : y + stripe_width, :] = rng.uniform(0.5, 1.0)
        else:
            for x in range(0, frames, stripe_width * 2):
                S[:, x : x + stripe_width] = rng.uniform(0.5, 1.0)

    # Clamp and normalize to [0,1]
    S -= S.min()
    if S.max() > 0:
        S /= S.max()

    return S.astype(np.float32)


if __name__ == "__main__":
    out_dir = Path("concepts_dataset_resnet_293")
    out_dir.mkdir(exist_ok=True)

    CREATE_ALL_CONCEPT_DIRS(out_dir)
