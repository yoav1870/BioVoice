from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from tcav_core.frame import FrameNormalizer


def first_concept_npy(concept_dir: Path) -> Optional[Path]:
    for file_path in sorted(concept_dir.glob("*.npy")):
        if file_path.name != "raw_energy.npy":
            return file_path
    return None


def infer_target_frames_from_concepts(concept_dirs: list[Path]) -> int:
    for concept_dir in concept_dirs:
        first_path = first_concept_npy(concept_dir)
        if first_path is not None:
            return int(np.load(first_path).shape[1])
    raise RuntimeError("Could not infer target frames from concept directories.")


def infer_module_device(module: Any) -> torch.device:
    if hasattr(module, "parameters"):
        try:
            return next(module.parameters()).device
        except StopIteration:
            pass
    if hasattr(module, "buffers"):
        try:
            return next(module.buffers()).device
        except StopIteration:
            pass
    return torch.device("cpu")


class ConceptNPYDataset(Dataset):
    def __init__(
        self,
        concept_dir: Path,
        n_mels: int,
        frame_normalizer: FrameNormalizer,
        seed: int,
        limit: Optional[int] = None,
        augment_on_oversample: bool = False,
        augment_time_shift_max: int = 0,
        augment_freq_shift_max: int = 0,
        augment_gain_min: float = 1.0,
        augment_gain_max: float = 1.0,
        augment_noise_std: float = 0.0,
    ) -> None:
        files = [
            path
            for path in sorted(concept_dir.glob("*.npy"))
            if path.name != "raw_energy.npy"
        ]
        if not files:
            raise RuntimeError(f"No concept .npy files found in {concept_dir}")

        self.files = files
        self.base_count = len(self.files)
        self.requested_count = self.base_count if limit is None else int(limit)
        if self.requested_count <= 0:
            raise RuntimeError(f"Invalid concept sample count: {self.requested_count}")

        self.augment_on_oversample = bool(augment_on_oversample)
        self.augment_time_shift_max = max(0, int(augment_time_shift_max))
        self.augment_freq_shift_max = max(0, int(augment_freq_shift_max))
        self.augment_gain_min = float(augment_gain_min)
        self.augment_gain_max = float(augment_gain_max)
        if self.augment_gain_max < self.augment_gain_min:
            self.augment_gain_min, self.augment_gain_max = (
                self.augment_gain_max,
                self.augment_gain_min,
            )
        self.augment_noise_std = max(0.0, float(augment_noise_std))

        if self.requested_count <= self.base_count:
            self.target_count = self.requested_count
            self.is_oversampled = False
        elif self.augment_on_oversample:
            self.target_count = self.requested_count
            self.is_oversampled = True
        else:
            self.target_count = self.base_count
            self.is_oversampled = False

        self.n_mels = int(n_mels)
        self.frame_normalizer = frame_normalizer
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.target_count

    def _augment_duplicate(
        self, x: torch.Tensor, rng: np.random.Generator
    ) -> torch.Tensor:
        if self.augment_time_shift_max > 0:
            shift_t = int(
                rng.integers(
                    -self.augment_time_shift_max, self.augment_time_shift_max + 1
                )
            )
            if shift_t != 0:
                x = torch.roll(x, shifts=shift_t, dims=-1)

        if self.augment_freq_shift_max > 0:
            shift_f = int(
                rng.integers(
                    -self.augment_freq_shift_max, self.augment_freq_shift_max + 1
                )
            )
            if shift_f != 0:
                x = torch.roll(x, shifts=shift_f, dims=-2)

        if self.augment_gain_max > self.augment_gain_min:
            gain = float(rng.uniform(self.augment_gain_min, self.augment_gain_max))
            x = x * gain

        if self.augment_noise_std > 0.0:
            noise_seed = int(rng.integers(0, 2**31 - 1))
            generator = torch.Generator().manual_seed(noise_seed)
            noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
            x = x + (noise * self.augment_noise_std)

        return x.contiguous()

    def __getitem__(self, idx: int) -> torch.Tensor:
        index = int(idx)
        file_index = index if index < self.base_count else (index % self.base_count)
        mel = np.load(self.files[file_index]).astype(np.float32)
        if int(mel.shape[0]) != self.n_mels:
            raise RuntimeError(
                f"{self.files[file_index].name}: expected {self.n_mels} mel bins, got {tuple(mel.shape)}"
            )
        x = torch.from_numpy(mel).unsqueeze(0)
        rng = np.random.default_rng(self.seed + index)
        x = self.frame_normalizer.normalize(x, rng)

        if (
            self.is_oversampled
            and index >= self.base_count
            and self.augment_on_oversample
        ):
            x = self._augment_duplicate(x, rng)
        return x


class RandomGaussianMelDataset(Dataset):
    def __init__(
        self, n_samples: int, n_mels: int, target_frames: int, seed: int
    ) -> None:
        self.n_samples = int(n_samples)
        self.n_mels = int(n_mels)
        self.target_frames = int(target_frames)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + int(idx))
        mel = torch.randn(
            self.n_mels,
            self.target_frames,
            generator=generator,
            dtype=torch.float32,
        )
        return mel.unsqueeze(0)


class ShuffledRealMelDataset(Dataset):
    def __init__(
        self,
        wav_paths: list[Path],
        n_samples: int,
        n_mels: int,
        frame_normalizer: FrameNormalizer,
        spec_fn,
        shuffle_time: bool,
        shuffle_freq: bool,
        seed: int,
        spec_device: Optional[Any] = None,
        target_sr: int = 16000,
        cache_size: int = 256,
    ) -> None:
        if not wav_paths:
            raise RuntimeError("ShuffledRealMelDataset requires at least one wav path.")
        self.wav_paths = [Path(path) for path in wav_paths]
        self.n_samples = int(n_samples)
        self.n_mels = int(n_mels)
        self.frame_normalizer = frame_normalizer
        self.spec_fn = spec_fn
        self.spec_device = (
            infer_module_device(self.spec_fn)
            if spec_device is None
            else torch.device(spec_device)
        )
        self.shuffle_time = bool(shuffle_time)
        self.shuffle_freq = bool(shuffle_freq)
        self.seed = int(seed)
        self.target_sr = int(target_sr)
        self.cache_size = int(cache_size)
        self._cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.n_samples

    def _load_base_mel(self, wav_path: Path, rng: np.random.Generator) -> torch.Tensor:
        key = wav_path.as_posix()
        if key in self._cache:
            return self._cache[key].clone()

        wav, sr = torchaudio.load(str(wav_path))
        wav = wav[:1, :].float()
        if int(sr) != self.target_sr:
            wav = torchaudio.functional.resample(wav, int(sr), self.target_sr)
        wav = wav.to(self.spec_device, non_blocking=True)

        with torch.no_grad():
            mel = self.spec_fn(wav)
        mel = mel.detach().cpu()

        mel = self.frame_normalizer.normalize(mel, rng).squeeze(0)
        if int(mel.shape[0]) != self.n_mels:
            raise RuntimeError(
                f"{wav_path.name}: expected {self.n_mels} mel bins, got {tuple(mel.shape)}"
            )

        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = mel.clone()
        return mel

    def __getitem__(self, idx: int) -> torch.Tensor:
        index = int(idx)
        rng = np.random.default_rng(self.seed + index)
        generator = torch.Generator().manual_seed(self.seed + index)

        source_idx = int(
            torch.randint(0, len(self.wav_paths), (1,), generator=generator).item()
        )
        mel = self._load_base_mel(self.wav_paths[source_idx], rng=rng)

        if self.shuffle_time:
            perm_t = torch.randperm(mel.shape[1], generator=generator)
            mel = mel[:, perm_t]
        if self.shuffle_freq:
            perm_f = torch.randperm(mel.shape[0], generator=generator)
            mel = mel[perm_f, :]

        return mel.unsqueeze(0).contiguous()
