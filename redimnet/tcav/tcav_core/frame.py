from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class FrameNormalizer:
    def __init__(self, target_frames: int, crop_mode: str, pad_mode: str) -> None:
        self.target_frames = int(target_frames)
        self.crop_mode = crop_mode.strip().lower()
        self.pad_mode = pad_mode.strip().lower()
        if self.target_frames <= 0:
            raise ValueError(f"target_frames must be > 0, got {self.target_frames}")

    def normalize(self, mel_3d: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        current_frames = int(mel_3d.shape[-1])
        if current_frames == self.target_frames:
            return mel_3d
        if current_frames > self.target_frames:
            return self._crop(mel_3d, current_frames, rng)
        return self._pad(mel_3d, current_frames, rng)

    def _crop(
        self,
        mel_3d: torch.Tensor,
        current_frames: int,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        span = current_frames - self.target_frames
        if self.crop_mode == "left":
            start = 0
        elif self.crop_mode == "right":
            start = span
        elif self.crop_mode == "center":
            start = span // 2
        elif self.crop_mode == "random":
            start = int(rng.integers(0, span + 1))
        else:
            raise ValueError(f"Unknown frame_crop_mode: {self.crop_mode}")
        return mel_3d[..., start : start + self.target_frames]

    def _pad(
        self,
        mel_3d: torch.Tensor,
        current_frames: int,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        pad = self.target_frames - current_frames
        if self.pad_mode == "right":
            left, right = 0, pad
        elif self.pad_mode == "left":
            left, right = pad, 0
        elif self.pad_mode == "center":
            left = pad // 2
            right = pad - left
        elif self.pad_mode == "random_side":
            left = int(rng.integers(0, pad + 1))
            right = pad - left
        else:
            raise ValueError(f"Unknown frame_pad_mode: {self.pad_mode}")
        return F.pad(mel_3d, (left, right), mode="constant", value=0.0)
