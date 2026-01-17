from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import ipywidgets as W
    from IPython.display import display
except Exception:  
    W = None
    display = None


def postprocess_like_redimnet(S_energy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    S = np.maximum(S_energy, 0.0)
    S = np.log(S + eps)
    S = S - S.mean(axis=1, keepdims=True)
    return S.astype(np.float32)


def _frames_from_seconds(seconds: float, hop_length: int, sample_rate: int) -> int:
    dt = hop_length / float(sample_rate)
    return int(round(seconds / dt))


def _clip_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


@dataclass
class SaveAugment:
    n_samples: int = 60
    noise_std: float = 0.02          # energy space
    max_time_shift: int = 6          # frames
    max_freq_shift: int = 2          # mel bins
    seed: Optional[int] = 42


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
        postprocess_fn: Callable[[np.ndarray], np.ndarray] = postprocess_like_redimnet,
        cmap: str = "hot",  # 0->black, mid->yellow, 1->white-ish
    ):
        self.mel_bins = int(mel_bins)
        self.frames = int(frames)
        self.seconds = seconds
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.postprocess_fn = postprocess_fn
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
        self._w_noise = None
        self._w_tshift = None
        self._w_fshift = None
        self._w_seed = None

    def _push_undo(self) -> None:
        self._undo.append(self.S.copy())
        self._redo.clear()

    def _apply_brush(self, y: int, x: int, value: float, radius: int, erase: bool) -> None:
        y = _clip_int(y, 0, self.mel_bins - 1)
        x = _clip_int(x, 0, self.frames - 1)
        r = max(0, int(radius))

        y0 = max(0, y - r)
        y1 = min(self.mel_bins - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.frames - 1, x + r)

        # circular-ish brush mask
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= r * r if r > 0 else (yy == y) & (xx == x)

        patch = self.S[y0 : y1 + 1, x0 : x1 + 1]
        if erase:
            patch[mask] = 0.0
        else:
            patch[mask] = np.float32(value)
        self.S[y0 : y1 + 1, x0 : x1 + 1] = patch

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
        self._im.set_data(self.S)
        self._fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        idx = self._event_to_indices(event)
        if idx is None:
            return
        self._pressed = True
        self._push_undo()

        y, x = idx
        val = float(self._w_intensity.value)
        r = int(self._w_brush.value)
        erase = bool(self._w_erase.value)
        self._apply_brush(y, x, val, r, erase)
        self._redraw()

    def _on_release(self, event) -> None:
        self._pressed = False

    def _on_move(self, event) -> None:
        if not self._pressed:
            return
        idx = self._event_to_indices(event)
        if idx is None:
            return

        y, x = idx
        val = float(self._w_intensity.value)
        r = int(self._w_brush.value)
        erase = bool(self._w_erase.value)
        self._apply_brush(y, x, val, r, erase)
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
        rng = np.random.default_rng(aug.seed)

        out: List[np.ndarray] = []
        for _ in range(int(aug.n_samples)):
            S = base.copy()

            if aug.noise_std and aug.noise_std > 0:
                S = S + rng.normal(0.0, aug.noise_std, size=S.shape).astype(np.float32)

            # time shift
            if aug.max_time_shift and aug.max_time_shift > 0:
                dt = int(rng.integers(-aug.max_time_shift, aug.max_time_shift + 1))
                if dt != 0:
                    S = np.roll(S, shift=dt, axis=1)

            # freq shift
            if aug.max_freq_shift and aug.max_freq_shift > 0:
                df = int(rng.integers(-aug.max_freq_shift, aug.max_freq_shift + 1))
                if df != 0:
                    S = np.roll(S, shift=df, axis=0)

            S = np.clip(S, 0.0, 1.0).astype(np.float32)
            out.append(S)

        return out

    def _save_preview_png(self, path: Path) -> None:
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(self.S, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap=self.cmap)
        fig.colorbar(im, ax=ax, fraction=0.025)
        ax.set_title("Manual Concept (energy 0..1)")
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

        np.save(concept_dir / "raw_energy.npy", self.S.astype(np.float32))

        self._save_preview_png(concept_dir / "preview.png")

        aug = SaveAugment(
            n_samples=int(self._w_save_n.value),
            noise_std=float(self._w_noise.value),
            max_time_shift=int(self._w_tshift.value),
            max_freq_shift=int(self._w_fshift.value),
            seed=int(self._w_seed.value) if self._w_seed.value is not None else None,
        )

        samples_energy = self._augment_samples(self.S, aug)
        for i, S_energy in enumerate(samples_energy, start=1):
            S_proc = self.postprocess_fn(S_energy)
            np.save(concept_dir / f"{i:06d}.npy", S_proc.astype(np.float32))

        meta = {
            "name": name,
            "mel_bins": self.mel_bins,
            "frames": self.frames,
            "seconds": self.seconds,
            "hop_length": self.hop_length,
            "sample_rate": self.sample_rate,
            "postprocess": getattr(self.postprocess_fn, "__name__", "custom"),
            "augment": {
                "n_samples": aug.n_samples,
                "noise_std": aug.noise_std,
                "max_time_shift": aug.max_time_shift,
                "max_freq_shift": aug.max_freq_shift,
                "seed": aug.seed,
            },
        }
        (concept_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"Saved concept to: {concept_dir}")
        print(f" - raw_energy.npy")
        print(f" - preview.png")
        print(f" - meta.json")
        print(f" - {aug.n_samples} x processed npy files")

    def launch(self) -> None:
        if W is None or display is None:
            raise RuntimeError(
                "ipywidgets is not available. Install ipywidgets + ipympl and run in Jupyter."
            )

        self._w_intensity = W.FloatSlider(
            value=1.0, min=0.0, max=1.0, step=0.05, description="value", readout_format=".2f"
        )
        self._w_brush = W.IntSlider(value=1, min=0, max=8, step=1, description="brush")
        self._w_erase = W.ToggleButton(value=False, description="erase", button_style="")
        self._w_grid = W.Checkbox(value=False, description="grid")
        self._w_grid.observe(self._toggle_grid, names="value")

        self._w_name = W.Text(value="", description="name", placeholder="e.g. long_rising_thin_manual")
        self._w_save_n = W.IntSlider(value=60, min=1, max=300, step=1, description="n")
        self._w_noise = W.FloatSlider(value=0.02, min=0.0, max=0.2, step=0.005, description="noise")
        self._w_tshift = W.IntSlider(value=6, min=0, max=40, step=1, description="tshift")
        self._w_fshift = W.IntSlider(value=2, min=0, max=12, step=1, description="fshift")
        self._w_seed = W.IntText(value=42, description="seed")

        btn_clear = W.Button(description="clear")
        btn_undo = W.Button(description="undo")
        btn_redo = W.Button(description="redo")
        btn_save = W.Button(description="save", button_style="success")

        btn_clear.on_click(self._do_clear)
        btn_undo.on_click(self._do_undo)
        btn_redo.on_click(self._do_redo)
        btn_save.on_click(self._do_save)

        controls_top = W.HBox([self._w_intensity, self._w_brush, self._w_erase, self._w_grid])
        controls_mid = W.HBox([btn_undo, btn_redo, btn_clear])
        controls_save = W.VBox(
            [
                W.HBox([self._w_name, btn_save]),
                W.HBox([self._w_save_n, self._w_noise]),
                W.HBox([self._w_tshift, self._w_fshift, self._w_seed]),
            ]
        )

        display(controls_top, controls_mid, controls_save)

        self._fig, self._ax = plt.subplots(figsize=(12, 3.5))
        self._im = self._ax.imshow(
            self.S,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap=self.cmap,
            interpolation="nearest",
        )
        self._fig.colorbar(self._im, ax=self._ax, fraction=0.025)
        self._ax.set_title("ConceptMaker (paint energy 0..1)")
        self._ax.set_xlabel("time frames")
        self._ax.set_ylabel("mel bins")

        if self.seconds is not None and self.frames > 1:
            ticks = np.linspace(0, self.frames - 1, num=min(9, self.frames), dtype=int)
            self._ax.set_xticks(ticks)
            sec_labels = np.linspace(0, self.seconds, num=len(ticks))
            self._ax.set_xticklabels([f"{s:.2f}s" for s in sec_labels])

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
    postprocess_fn: Callable[[np.ndarray], np.ndarray] = postprocess_like_redimnet,
) -> ConceptMaker:
    if frames is None:
        if seconds is None or hop_length is None or sample_rate is None:
            raise ValueError("Provide either frames=... OR (seconds, hop_length, sample_rate).")
        frames = _frames_from_seconds(seconds, hop_length, sample_rate)

    cm = ConceptMaker(
        mel_bins=mel_bins,
        frames=frames,
        seconds=seconds,
        hop_length=hop_length,
        sample_rate=sample_rate,
        out_dir=out_dir,
        postprocess_fn=postprocess_fn,
    )
    cm.launch()
    return cm
