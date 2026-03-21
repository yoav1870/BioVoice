from __future__ import annotations
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
from matplotlib.colors import Normalize


import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import ipywidgets as W
    from IPython.display import display
except Exception:
    W = None
    display = None


def _frames_from_seconds(seconds: float, hop_length: int, sample_rate: int) -> int:
    dt = hop_length / float(sample_rate)
    return int(round(seconds / dt))


def _clip_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


@dataclass
class SaveAugment:
    n_samples: int = 60
    seed: Optional[int] = 42
    cover_thr: float = 1e-6  # pixels <= thr considered empty
    cover_rows: int = 3
    cover_cols: int = 3
    amp_jitter: float = 0.15
    thickness_jitter: float = 0.08


class ConceptMaker:
    def __init__(
        self,
        *,
        mel_bins: int,
        frames: int,
        seconds: Optional[float] = None,
        hop_length: Optional[int] = None,
        sample_rate: Optional[int] = None,
        out_dir: Path,
        cmap: str = "inferno",
        db_max: float = 12.0,
        save_raw_energy: bool = False,
    ):
        self.mel_bins = int(mel_bins)
        self.frames = int(frames)
        self.seconds = seconds
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.db_max = float(db_max)
        self.save_raw_energy = bool(save_raw_energy)

        self.cmap = cmap

        self.S = np.zeros((self.mel_bins, self.frames), dtype=np.float32)

        self._undo: List[np.ndarray] = []
        self._redo: List[np.ndarray] = []

        self._fig = None
        self._ax = None
        self._im = None
        self._pressed = False

        self._w_intensity = None
        self._w_brush = None
        self._w_erase = None
        self._w_grid = None
        self._w_name = None
        self._w_save_n = None
        self._w_seed = None
        self._w_amp_jitter = None
        self._w_thickness_jitter = None
        self._w_feather = None
        self._w_tool = None
        self._w_blend = None
        self._tool_points: List[Tuple[int, int]] = []  # list of (y,x) clicks

        self._raw_norm = None
        self._cbar = None

    def _push_undo(self) -> None:
        self._undo.append(self.S.copy())
        self._redo.clear()

    def _apply_brush(
        self, y: int, x: int, value: float, radius: int, erase: bool
    ) -> None:
        y = _clip_int(y, 0, self.mel_bins - 1)
        x = _clip_int(x, 0, self.frames - 1)
        r = max(0, int(radius))

        y0 = max(0, y - r)
        y1 = min(self.mel_bins - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.frames - 1, x + r)

        patch = self.S[y0 : y1 + 1, x0 : x1 + 1]

        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        dy = (yy - y).astype(np.float32)
        dx = (xx - x).astype(np.float32)
        dist2 = dy * dy + dx * dx

        if r == 0:
            w = np.ones_like(patch, dtype=np.float32)
        else:
            base = float(self._w_feather.value) if self._w_feather is not None else 1.5
            # scale sigma with radius so "brush" actually controls thickness
            sigma = max(1e-3, base * max(1.0, np.sqrt(r)))

            w = np.exp(-0.5 * dist2 / (sigma * sigma)).astype(np.float32)
            w *= (dist2 <= (r * r)).astype(np.float32)

        if erase:
            patch *= 1.0 - w
        else:
            target = np.float32(value) * w
            blend = self._w_blend.value if self._w_blend is not None else "max"
            if blend == "add":
                patch[:] = patch + target
            else:
                patch[:] = np.maximum(patch, target)

        self.S[y0 : y1 + 1, x0 : x1 + 1] = np.clip(patch, 0.0, 1.0).astype(np.float32)

    def _stamp(self, y: int, x: int) -> None:
        val_db = float(self._w_intensity.value)  # 0..12
        val = val_db / self.db_max  # 0..1 internal
        r = int(self._w_brush.value)
        erase = bool(self._w_erase.value)
        self._apply_brush(y, x, val, r, erase)

    def _draw_line(self, p0: Tuple[int, int], p1: Tuple[int, int]) -> None:
        (y0, x0), (y1, x1) = p0, p1
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        n = max(dy, dx, 1)

        ys = np.round(np.linspace(y0, y1, num=n + 1)).astype(int)
        xs = np.round(np.linspace(x0, x1, num=n + 1)).astype(int)

        for y, x in zip(ys, xs):
            self._stamp(int(y), int(x))

    def _draw_quadratic_bezier(
        self, p0: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int]
    ) -> None:
        (y0, x0), (y1, x1), (y2, x2) = p0, p1, p2

        span = max(abs(y2 - y0), abs(x2 - x0), abs(y1 - y0), abs(x1 - x0), 1)
        n = int(span * 2)  # denser than a straight line

        t = np.linspace(0.0, 1.0, num=n + 1, dtype=np.float32)
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t**2 * y2
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t**2 * x2

        ys = np.round(y).astype(int)
        xs = np.round(x).astype(int)

        last = None
        for yi, xi in zip(ys, xs):
            pt = (int(yi), int(xi))
            if pt != last:
                self._stamp(pt[0], pt[1])
                last = pt

    def _event_to_indices(self, event) -> Optional[Tuple[int, int]]:
        if event.inaxes != self._ax:
            return None
        if event.xdata is None or event.ydata is None:
            return None

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if not (0 <= x < self.frames and 0 <= y < self.mel_bins):
            return None
        return (y, x)

    def _redraw(self) -> None:
        if self._im is None:
            return
        if self._raw_norm is not None:
            self._im.set_norm(self._raw_norm)
        self._im.set_data(self.S * self.db_max)
        if self._cbar is not None:
            self._cbar.set_label(f"dB (0..{self.db_max:g})")
            self._cbar.update_normal(self._im)
        self._fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        idx = self._event_to_indices(event)
        if idx is None:
            return

        tool = self._w_tool.value if self._w_tool is not None else "brush"

        if tool == "brush":
            self._pressed = True
            self._push_undo()
            y, x = idx
            self._stamp(y, x)
            self._redraw()
            return

        y, x = idx
        self._tool_points.append((y, x))

        if tool == "line" and len(self._tool_points) == 2:
            self._push_undo()
            self._draw_line(self._tool_points[0], self._tool_points[1])
            self._tool_points.clear()
            self._redraw()
            return

        if tool == "curve" and len(self._tool_points) == 3:
            self._push_undo()
            self._draw_quadratic_bezier(
                self._tool_points[0], self._tool_points[1], self._tool_points[2]
            )
            self._tool_points.clear()
            self._redraw()
            return

    def _on_release(self, event) -> None:
        self._pressed = False

    def _on_move(self, event) -> None:
        if not self._pressed:
            return
        if self._w_tool is not None and self._w_tool.value != "brush":
            return

        idx = self._event_to_indices(event)
        if idx is None:
            return
        y, x = idx
        self._stamp(y, x)
        self._redraw()

    def _do_clear(self, _btn=None) -> None:
        self._push_undo()
        self.S.fill(0.0)
        self._redraw()

    def _do_undo(self, _btn=None) -> None:
        if not self._undo:
            return
        self._redo.append(self.S.copy())
        self.S = self._undo.pop()
        self._redraw()

    def _do_redo(self, _btn=None) -> None:
        if not self._redo:
            return
        self._undo.append(self.S.copy())
        self.S = self._redo.pop()
        self._redraw()

    def _toggle_grid(self, change) -> None:
        if self._ax is None:
            return
        show = bool(change["new"])
        self._ax.grid(show, which="both", linewidth=0.3, alpha=0.35)
        self._fig.canvas.draw_idle()

    def _augment_samples(self, base: np.ndarray, aug: SaveAugment) -> List[np.ndarray]:
        return self._grid_cover_samples(base, aug)

    def _extract_nonzero_patch(
        self, base: np.ndarray, thr: float
    ) -> Optional[np.ndarray]:
        mask = base > thr
        if not mask.any():
            return None
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        return base[y0 : y1 + 1, x0 : x1 + 1].copy()

    def _split_placement_axis(self, max_pos: int, n_cells: int) -> List[Tuple[int, int]]:
        n = max(1, int(n_cells))
        total = int(max_pos) + 1
        out: List[Tuple[int, int]] = []
        for i in range(n):
            lo = int(np.floor((i * total) / n))
            hi = int(np.floor(((i + 1) * total) / n) - 1)
            lo = _clip_int(lo, 0, max_pos)
            hi = _clip_int(hi, 0, max_pos)
            if hi < lo:
                hi = lo
            out.append((lo, hi))
        return out

    def _rescale_patch_centered(self, patch: np.ndarray, scale: float) -> np.ndarray:
        ph, pw = patch.shape
        if ph <= 1 or pw <= 1:
            return patch.astype(np.float32, copy=True)

        s = max(1e-3, float(scale))
        if abs(s - 1.0) < 1e-6:
            return patch.astype(np.float32, copy=True)

        cy = 0.5 * (ph - 1)
        cx = 0.5 * (pw - 1)
        ys = (np.arange(ph, dtype=np.float32) - cy) / s + cy
        xs = (np.arange(pw, dtype=np.float32) - cx) / s + cx

        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        y1 = y0 + 1
        x1 = x0 + 1

        wy = (ys - y0).astype(np.float32)[:, None]
        wx = (xs - x0).astype(np.float32)[None, :]

        y0c = np.clip(y0, 0, ph - 1)
        y1c = np.clip(y1, 0, ph - 1)
        x0c = np.clip(x0, 0, pw - 1)
        x1c = np.clip(x1, 0, pw - 1)

        Ia = patch[np.ix_(y0c, x0c)]
        Ib = patch[np.ix_(y0c, x1c)]
        Ic = patch[np.ix_(y1c, x0c)]
        Id = patch[np.ix_(y1c, x1c)]

        out = (
            (1.0 - wy) * (1.0 - wx) * Ia
            + (1.0 - wy) * wx * Ib
            + wy * (1.0 - wx) * Ic
            + wy * wx * Id
        ).astype(np.float32)

        valid_y = (ys >= 0.0) & (ys <= float(ph - 1))
        valid_x = (xs >= 0.0) & (xs <= float(pw - 1))
        out *= (valid_y[:, None] & valid_x[None, :]).astype(np.float32)
        return out

    def _grid_cover_samples(self, base: np.ndarray, aug: SaveAugment) -> List[np.ndarray]:
        patch = self._extract_nonzero_patch(base, aug.cover_thr)
        if patch is None:
            return [base.astype(np.float32)]  # nothing drawn -> just one sample

        ph, pw = patch.shape
        y_max = self.mel_bins - ph
        x_max = self.frames - pw
        if y_max < 0 or x_max < 0:
            return [
                base.astype(np.float32)
            ]  # patch bigger than canvas (shouldn't happen)

        rng = np.random.default_rng(aug.seed)
        target = max(1, int(aug.n_samples))
        rows = max(1, int(aug.cover_rows))
        cols = max(1, int(aug.cover_cols))
        amp_jitter = max(0.0, float(aug.amp_jitter))
        thickness_jitter = max(0.0, float(aug.thickness_jitter))

        y_ranges = self._split_placement_axis(y_max, rows)
        x_ranges = self._split_placement_axis(x_max, cols)
        cells = [(yr, xr) for yr in y_ranges for xr in x_ranges]
        n_cells = len(cells)

        counts = [target // n_cells] * n_cells
        remainder = target % n_cells
        if remainder > 0:
            for idx in rng.permutation(n_cells)[:remainder]:
                counts[int(idx)] += 1

        out: List[np.ndarray] = []
        for cell_idx, count in enumerate(counts):
            if count <= 0:
                continue

            (y0, y1), (x0, x1) = cells[cell_idx]
            for _ in range(count):
                y = int(rng.integers(y0, y1 + 1)) if y1 > y0 else y0
                x = int(rng.integers(x0, x1 + 1)) if x1 > x0 else x0

                amp_scale = 1.0
                if amp_jitter > 0.0:
                    amp_scale += float(rng.uniform(-amp_jitter, amp_jitter))
                    amp_scale = max(0.0, amp_scale)

                thick_scale = 1.0
                if thickness_jitter > 0.0:
                    thick_scale += float(rng.uniform(-thickness_jitter, thickness_jitter))

                p = self._rescale_patch_centered(patch, thick_scale)
                p = np.clip(p * amp_scale, 0.0, 1.0).astype(np.float32)

                S = np.zeros((self.mel_bins, self.frames), dtype=np.float32)
                S[y : y + ph, x : x + pw] = p
                out.append(S)

        return out

    def _save_preview_png(self, path: Path) -> None:
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(
            self.S * self.db_max,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=self.db_max,
            cmap=self.cmap,
        )
        fig.colorbar(im, ax=ax, fraction=0.025)
        ax.set_title(f"Manual Concept (0..{self.db_max:g} dB)")
        ax.set_xlabel("frames")
        ax.set_ylabel("mel bins")
        fig.tight_layout()
        fig.savefig(path.as_posix(), dpi=160)
        plt.close(fig)

    def _do_save(self, _btn=None) -> None:
        name = (self._w_name.value or "").strip()
        if not name:
            raise ValueError("Concept name is empty")

        concept_dir = self.out_dir / name
        concept_dir.mkdir(parents=True, exist_ok=True)

        save_raw = bool(self.save_raw_energy)
        if save_raw:
            np.save(concept_dir / "raw_energy.npy", self.S.astype(np.float32))

        self._save_preview_png(concept_dir / "preview.png")

        aug = SaveAugment(
            n_samples=int(self._w_save_n.value),
            seed=int(self._w_seed.value) if self._w_seed.value is not None else None,
            cover_rows=3,
            cover_cols=3,
            amp_jitter=float(self._w_amp_jitter.value),
            thickness_jitter=float(self._w_thickness_jitter.value),
        )

        samples_energy = self._augment_samples(self.S, aug)
        for i, S_energy in enumerate(samples_energy, start=1):
            S_db = (S_energy * self.db_max).astype(np.float32)
            np.save(concept_dir / f"{i:06d}.npy", S_db)

        meta = {
            "name": name,
            "mel_bins": self.mel_bins,
            "frames": self.frames,
            "seconds": self.seconds,
            "hop_length": self.hop_length,
            "sample_rate": self.sample_rate,
            "save_raw_energy": save_raw,
            "saved_scale": f"0..{self.db_max:g} dB",
            "augment": {
                "n_samples": aug.n_samples,
                "seed": aug.seed,
                "cover_thr": aug.cover_thr,
                "cover_rows": aug.cover_rows,
                "cover_cols": aug.cover_cols,
                "amp_jitter": aug.amp_jitter,
                "thickness_jitter": aug.thickness_jitter,
            },
        }
        (concept_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"Saved concept to: {concept_dir}")
        if save_raw:
            print(f" - raw_energy.npy")
        print(f" - preview.png")
        print(f" - meta.json")
        print(f" - {aug.n_samples} x saved npy files (dB)")

    def launch(self) -> None:
        if W is None or display is None:
            raise RuntimeError(
                "ipywidgets is not available. Install ipywidgets + ipympl and run in Jupyter."
            )

        self._w_intensity = W.FloatSlider(
            value=self.db_max,
            min=0.0,
            max=self.db_max,
            step=0.25,
            description="dB",
            readout_format=".2f",
        )
        self._w_brush = W.IntSlider(value=1, min=0, max=8, step=1, description="brush")
        self._w_erase = W.ToggleButton(
            value=False, description="erase", button_style=""
        )
        self._w_grid = W.Checkbox(value=False, description="grid")
        self._w_grid.observe(self._toggle_grid, names="value")
        self._w_tool = W.Dropdown(
            options=[
                ("Brush", "brush"),
                ("Line (2 clicks)", "line"),
                ("Curve (3 clicks)", "curve"),
            ],
            value="brush",
            description="tool",
        )
        self._w_blend = W.Dropdown(
            options=[("Max (crisp)", "max"), ("Add (layer)", "add")],
            value="max",
            description="blend",
        )
        self._w_name = W.Text(
            value="", description="name", placeholder="e.g. long_rising_thin_manual"
        )
        self._w_save_n = W.IntSlider(value=60, min=1, max=300, step=1, description="n")
        self._w_seed = W.IntText(value=42, description="seed")
        self._w_amp_jitter = W.FloatSlider(
            value=0.15, min=0.0, max=0.5, step=0.01, description="amp jit"
        )
        self._w_thickness_jitter = W.FloatSlider(
            value=0.08, min=0.0, max=0.25, step=0.01, description="thick jit"
        )
        self._w_feather = W.FloatSlider(
            value=1.5, min=0.2, max=8.0, step=0.1, description="feather"
        )

        btn_clear = W.Button(description="clear")
        btn_undo = W.Button(description="undo")
        btn_redo = W.Button(description="redo")
        btn_save = W.Button(description="save", button_style="success")

        btn_clear.on_click(self._do_clear)
        btn_undo.on_click(self._do_undo)
        btn_redo.on_click(self._do_redo)
        btn_save.on_click(self._do_save)

        controls_top = W.HBox(
            [
                self._w_tool,
                self._w_intensity,
                self._w_brush,
                self._w_feather,
                self._w_blend,
                self._w_erase,
                self._w_grid,
            ]
        )
        controls_mid = W.HBox([btn_undo, btn_redo, btn_clear])
        controls_save = W.VBox(
            [
                W.HBox([self._w_name, btn_save]),
                W.HBox([self._w_save_n, self._w_seed]),
                W.HBox([self._w_amp_jitter, self._w_thickness_jitter]),
            ]
        )

        display(controls_top, controls_mid, controls_save)

        self._fig, self._ax = plt.subplots(figsize=(12, 3.5))
        self._raw_norm = Normalize(vmin=0.0, vmax=self.db_max)

        self._im = self._ax.imshow(
            self.S * self.db_max,
            origin="lower",
            aspect="auto",
            cmap=self.cmap,
            norm=self._raw_norm,
            interpolation="nearest",
        )
        self._cbar = self._fig.colorbar(self._im, ax=self._ax, fraction=0.025)
        self._cbar.set_label(f"dB (0..{self.db_max:g})")
        self._ax.set_title(f"ConceptMaker (paint 0..{self.db_max:g} dB)")
        self._ax.set_xlabel("time frames")
        self._ax.set_ylabel("mel bins")

        if self.seconds is not None and self.frames > 1:
            dt = self.seconds / float(self.frames - 1)
            ticks = np.linspace(0, self.frames - 1, num=min(9, self.frames), dtype=int)
            self._ax.set_xticks(ticks)
            self._ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{x * dt:.2f}s")
            )
        self._ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        self._ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        self._fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_move)

        plt.show()


def launch_concept_maker(
    *,
    mel_bins: int,
    out_dir: Path,
    frames: Optional[int] = None,
    seconds: Optional[float] = None,
    hop_length: Optional[int] = None,
    sample_rate: Optional[int] = None,
    save_raw_energy: bool = False,
) -> ConceptMaker:
    if frames is None:
        if seconds is None or hop_length is None or sample_rate is None:
            raise ValueError(
                "Provide either frames=... OR (seconds, hop_length, sample_rate)."
            )
        frames = _frames_from_seconds(seconds, hop_length, sample_rate)

    cm = ConceptMaker(
        mel_bins=mel_bins,
        frames=frames,
        seconds=seconds,
        hop_length=hop_length,
        sample_rate=sample_rate,
        out_dir=out_dir,
        save_raw_energy=save_raw_energy,
    )
    cm.launch()
    return cm


def _load_preprocess_params():
    try:
        from PreprocessParams import (
            FREQUENCY_BIN_COUNT,
            TARGET_FRAMES_PAD,
            SAMPLE_RATE,
            HOP_LENGTH,
            MAX_SPECTOGRAM_DURATION_IN_SECONDS,
        )
    except Exception as exc:
        raise RuntimeError(
            "PreprocessParams is not importable. Run from the concept/ folder "
            "or add it to sys.path."
        ) from exc

    return {
        "mel_bins": int(FREQUENCY_BIN_COUNT),
        "frames": int(TARGET_FRAMES_PAD),
        "sample_rate": int(SAMPLE_RATE),
        "hop_length": int(HOP_LENGTH),
        "seconds": float(MAX_SPECTOGRAM_DURATION_IN_SECONDS),
    }


def launch_concept_maker_like_concepts_creation(
    *,
    out_dir: Path,
    save_raw_energy: bool = False,
) -> ConceptMaker:
    """
    Launch ConceptMaker with the same mel/frames settings used in concepts_creation.py.
    """
    params = _load_preprocess_params()
    return launch_concept_maker(
        mel_bins=params["mel_bins"],
        frames=params["frames"],
        seconds=params["seconds"],
        hop_length=params["hop_length"],
        sample_rate=params["sample_rate"],
        out_dir=out_dir,
        save_raw_energy=save_raw_energy,
    )
