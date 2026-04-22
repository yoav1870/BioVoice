from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Canvas:
    n_mels: int = 80
    n_frames: int = 200


@dataclass(frozen=True)
class GenConfig:
    seed: int = 1337
    samples_per_concept: int = 200
    nuisance_mix_prob: float = 0.30
    nuisance_mix_scale: float = 0.20
    save_preview_every: int = 25


CANVAS = Canvas()

CONCEPT_NAMES = [
    "wide_high_band",
    "wide_mid_band",
    "wide_lowmid_band",
    "thin_high_band",
    "thin_mid_band",
    "band_stack_2",
    "band_stack_3",
    "fragmented_band_segments",
    "local_patch_cluster",
    "vertical_transient",
    "rising_or_falling_slanted_band",
    "negative_random_smooth",
]


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def _blur2d(arr: np.ndarray, sigma_mel: float, sigma_time: float) -> np.ndarray:
    """Separable Gaussian blur without scipy dependency."""
    out = arr.astype(np.float32, copy=True)
    km = _gaussian_kernel1d(sigma_mel)
    kt = _gaussian_kernel1d(sigma_time)

    # Convolve along mel axis (rows)
    out = np.apply_along_axis(lambda m: np.convolve(m, km, mode="same"), axis=0, arr=out)
    # Convolve along time axis (cols)
    out = np.apply_along_axis(lambda m: np.convolve(m, kt, mode="same"), axis=1, arr=out)
    return out.astype(np.float32)


def _new_canvas() -> np.ndarray:
    return np.zeros((CANVAS.n_mels, CANVAS.n_frames), dtype=np.float32)


def _draw_horizontal_band(
    s: np.ndarray,
    mel_center: int,
    thickness: int,
    t0: int,
    t1: int,
    amp: float,
) -> None:
    h = max(1, thickness // 2)
    m0 = _clamp(mel_center - h, 0, CANVAS.n_mels - 1)
    m1 = _clamp(mel_center + h, 0, CANVAS.n_mels - 1)
    t0 = _clamp(t0, 0, CANVAS.n_frames - 1)
    t1 = _clamp(t1, 0, CANVAS.n_frames)
    if t1 <= t0:
        return
    s[m0:m1, t0:t1] += amp


def _draw_rect(
    s: np.ndarray,
    mel0: int,
    mel_h: int,
    time0: int,
    time_w: int,
    amp: float,
) -> None:
    m0 = _clamp(mel0, 0, CANVAS.n_mels - 1)
    m1 = _clamp(m0 + mel_h, 0, CANVAS.n_mels)
    t0 = _clamp(time0, 0, CANVAS.n_frames - 1)
    t1 = _clamp(t0 + time_w, 0, CANVAS.n_frames)
    if m1 <= m0 or t1 <= t0:
        return
    s[m0:m1, t0:t1] += amp


def _draw_slanted_band(
    s: np.ndarray,
    mel_start: int,
    mel_end: int,
    thickness: int,
    t0: int,
    duration: int,
    amp: float,
) -> None:
    t1 = _clamp(t0 + duration, 0, CANVAS.n_frames)
    if t1 <= t0:
        return
    steps = t1 - t0
    centers = np.linspace(mel_start, mel_end, steps)
    h = max(1, thickness // 2)
    for i, mc in enumerate(centers):
        t = t0 + i
        m0 = _clamp(int(round(mc)) - h, 0, CANVAS.n_mels - 1)
        m1 = _clamp(int(round(mc)) + h, 0, CANVAS.n_mels - 1)
        s[m0:m1, t : t + 1] += amp


def _finalize_sample(
    s: np.ndarray,
    rng: np.random.Generator,
    blur_mel: tuple[float, float] = (0.8, 2.2),
    blur_time: tuple[float, float] = (0.8, 2.2),
) -> np.ndarray:
    sig_m = float(rng.uniform(*blur_mel))
    sig_t = float(rng.uniform(*blur_time))
    s = _blur2d(s, sig_m, sig_t)
    s = np.clip(s, 0.0, None)
    mx = float(np.max(s))
    if mx > 1e-8:
        s /= mx
    return s.astype(np.float32)


def _concept_negative_random_smooth(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    n_blobs = int(rng.integers(3, 11))
    for _ in range(n_blobs):
        h = int(rng.integers(5, 19))
        w = int(rng.integers(8, 31))
        m0 = int(rng.integers(0, max(1, CANVAS.n_mels - h)))
        t0 = int(rng.integers(0, max(1, CANVAS.n_frames - w)))
        amp = float(rng.uniform(0.35, 1.0))
        _draw_rect(s, m0, h, t0, w, amp)
    return _finalize_sample(s, rng, (1.0, 2.8), (1.0, 2.8))


def _concept_wide_high_band(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    c = int(rng.integers(56, 67))
    thick = int(rng.integers(8, 15))
    length = int(rng.integers(120, 191))
    t0 = int(rng.integers(0, max(1, CANVAS.n_frames - length)))
    _draw_horizontal_band(s, c, thick, t0, t0 + length, float(rng.uniform(0.6, 1.0)))
    return _finalize_sample(s, rng)


def _concept_wide_mid_band(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    c = int(rng.integers(46, 57))
    thick = int(rng.integers(8, 15))
    length = int(rng.integers(120, 191))
    t0 = int(rng.integers(0, max(1, CANVAS.n_frames - length)))
    _draw_horizontal_band(s, c, thick, t0, t0 + length, float(rng.uniform(0.6, 1.0)))
    return _finalize_sample(s, rng)


def _concept_wide_lowmid_band(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    c = int(rng.integers(24, 37))
    thick = int(rng.integers(8, 15))
    length = int(rng.integers(100, 181))
    t0 = int(rng.integers(0, max(1, CANVAS.n_frames - length)))
    _draw_horizontal_band(s, c, thick, t0, t0 + length, float(rng.uniform(0.6, 1.0)))
    return _finalize_sample(s, rng)


def _concept_thin_high_band(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    n = int(rng.integers(1, 3))
    for _ in range(n):
        c = int(rng.integers(58, 69))
        thick = int(rng.integers(3, 7))
        length = int(rng.integers(80, 181))
        t0 = int(rng.integers(0, max(1, CANVAS.n_frames - length)))
        _draw_horizontal_band(s, c, thick, t0, t0 + length, float(rng.uniform(0.6, 1.0)))
    return _finalize_sample(s, rng)


def _concept_thin_mid_band(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    n = int(rng.integers(1, 3))
    for _ in range(n):
        c = int(rng.integers(48, 59))
        thick = int(rng.integers(3, 7))
        length = int(rng.integers(80, 181))
        t0 = int(rng.integers(0, max(1, CANVAS.n_frames - length)))
        _draw_horizontal_band(s, c, thick, t0, t0 + length, float(rng.uniform(0.6, 1.0)))
    return _finalize_sample(s, rng)


def _concept_band_stack_2(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    c1 = int(rng.integers(24, 37))
    c2 = int(rng.integers(54, 67))
    for c in (c1, c2):
        thick = int(rng.integers(4, 11))
        length = int(rng.integers(90, 181))
        t0 = int(rng.integers(0, max(1, CANVAS.n_frames - length)))
        _draw_horizontal_band(s, c, thick, t0, t0 + length, float(rng.uniform(0.55, 1.0)))
    return _finalize_sample(s, rng)


def _concept_band_stack_3(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    base = int(rng.integers(18, 32))
    gap1 = int(rng.integers(10, 25))
    gap2 = int(rng.integers(10, 25))
    centers = [base, _clamp(base + gap1, 0, 79), _clamp(base + gap1 + gap2, 0, 79)]
    for c in centers:
        thick = int(rng.integers(4, 10))
        length = int(rng.integers(70, 161))
        t0 = int(rng.integers(0, max(1, CANVAS.n_frames - length)))
        _draw_horizontal_band(s, c, thick, t0, t0 + length, float(rng.uniform(0.55, 1.0)))
    return _finalize_sample(s, rng)


def _concept_fragmented_band_segments(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    center = int(rng.integers(24, 37)) if rng.random() < 0.5 else int(rng.integers(50, 67))
    thick = int(rng.integers(5, 13))
    n_seg = int(rng.integers(3, 8))
    t = int(rng.integers(0, 20))
    for _ in range(n_seg):
        seg_len = int(rng.integers(18, 56))
        _draw_horizontal_band(s, center + int(rng.integers(-2, 3)), thick, t, t + seg_len, float(rng.uniform(0.55, 1.0)))
        t += seg_len + int(rng.integers(8, 31))
        if t >= CANVAS.n_frames:
            break
    return _finalize_sample(s, rng)


def _concept_local_patch_cluster(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    n = int(rng.integers(2, 7))
    for _ in range(n):
        h = int(rng.integers(6, 17))
        w = int(rng.integers(10, 36))
        zone_low = rng.random() < 0.5
        m0 = int(rng.integers(20, 36)) if zone_low else int(rng.integers(50, 67))
        t0 = int(rng.integers(0, max(1, CANVAS.n_frames - w)))
        _draw_rect(s, m0, h, t0, w, float(rng.uniform(0.5, 1.0)))
    return _finalize_sample(s, rng, (0.9, 2.5), (0.9, 2.5))


def _concept_vertical_transient(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    n = int(rng.integers(1, 5))
    for _ in range(n):
        w = int(rng.integers(3, 11))
        h = int(rng.integers(18, 51))
        m0 = int(rng.integers(0, max(1, CANVAS.n_mels - h)))
        t0 = int(rng.integers(0, max(1, CANVAS.n_frames - w)))
        _draw_rect(s, m0, h, t0, w, float(rng.uniform(0.55, 1.0)))
    return _finalize_sample(s, rng, (0.8, 1.8), (0.8, 1.8))


def _concept_rising_or_falling_slanted_band(rng: np.random.Generator) -> np.ndarray:
    s = _new_canvas()
    dur = int(rng.integers(50, 141))
    t0 = int(rng.integers(0, max(1, CANVAS.n_frames - dur)))
    shift = int(rng.integers(6, 21))
    base = int(rng.integers(20, 55))
    direction = 1 if rng.random() < 0.5 else -1
    m0 = _clamp(base, 0, 79)
    m1 = _clamp(base + direction * shift, 0, 79)
    thick = int(rng.integers(4, 11))
    _draw_slanted_band(s, m0, m1, thick, t0, dur, float(rng.uniform(0.55, 1.0)))
    return _finalize_sample(s, rng)


GENERATORS = {
    "wide_high_band": _concept_wide_high_band,
    "wide_mid_band": _concept_wide_mid_band,
    "wide_lowmid_band": _concept_wide_lowmid_band,
    "thin_high_band": _concept_thin_high_band,
    "thin_mid_band": _concept_thin_mid_band,
    "band_stack_2": _concept_band_stack_2,
    "band_stack_3": _concept_band_stack_3,
    "fragmented_band_segments": _concept_fragmented_band_segments,
    "local_patch_cluster": _concept_local_patch_cluster,
    "vertical_transient": _concept_vertical_transient,
    "rising_or_falling_slanted_band": _concept_rising_or_falling_slanted_band,
    "negative_random_smooth": _concept_negative_random_smooth,
}


def _mix_nuisance(base: np.ndarray, rng: np.random.Generator, cfg: GenConfig) -> np.ndarray:
    if rng.random() > cfg.nuisance_mix_prob:
        return base
    nuisance = _concept_negative_random_smooth(rng)
    mixed = np.clip(base + cfg.nuisance_mix_scale * nuisance, 0.0, 1.0)
    mx = float(np.max(mixed))
    if mx > 1e-8:
        mixed /= mx
    return mixed.astype(np.float32)


def _save_preview_stack(out_path: Path, arrs: list[np.ndarray]) -> None:
    """Save a compact grid preview as .npy stack for quick checks without matplotlib."""
    if not arrs:
        return
    stack = np.stack(arrs, axis=0).astype(np.float32)
    np.save(out_path, stack)


def generate_all(out_root: Path, cfg: GenConfig) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "canvas": asdict(CANVAS),
        "config": asdict(cfg),
        "concepts": CONCEPT_NAMES,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rng_master = np.random.default_rng(cfg.seed)

    for concept in CONCEPT_NAMES:
        concept_dir = out_root / concept
        concept_dir.mkdir(parents=True, exist_ok=True)
        print(f"[concept] {concept} -> {concept_dir}")

        preview: list[np.ndarray] = []
        gen_fn = GENERATORS[concept]

        for i in range(cfg.samples_per_concept):
            # deterministic sub-seed per concept/sample
            sub_seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(sub_seed)

            s = gen_fn(rng)
            if concept != "negative_random_smooth":
                s = _mix_nuisance(s, rng, cfg)

            # tiny random gain jitter
            gain = float(rng.uniform(0.85, 1.15))
            s = np.clip(s * gain, 0.0, 1.0).astype(np.float32)

            np.save(concept_dir / f"{i:06d}.npy", s)

            if i % cfg.save_preview_every == 0:
                preview.append(s)

            if (i + 1) % 50 == 0 or i == cfg.samples_per_concept - 1:
                print(f"  - {i + 1}/{cfg.samples_per_concept}", flush=True)

        _save_preview_stack(concept_dir / "_preview_stack.npy", preview[:16])

    print("Done. Generated all concepts.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 12 Layer3 multisystem concepts.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("concept/layer3_multisystem_concepts/concepts"),
        help="Output directory for concept folders.",
    )
    parser.add_argument(
        "--samples-per-concept",
        type=int,
        default=200,
        help="Number of .npy samples per concept.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenConfig(seed=args.seed, samples_per_concept=args.samples_per_concept)
    print(f"Output: {args.out_dir.resolve()}")
    print(f"Config: {cfg}")
    generate_all(args.out_dir, cfg)


if __name__ == "__main__":
    main()

