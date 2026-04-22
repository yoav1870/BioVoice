# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from .common import module_name_in_model


def concept_dir(concept_root: Path, name: str) -> Path:
    return concept_root / name


def concept_npy_paths(concept_root: Path, name: str) -> list[Path]:
    filenames = sorted(concept_dir(concept_root, name).glob("*.npy"))
    if not filenames:
        raise FileNotFoundError(f"No .npy files found for concept: {name}")
    return filenames


def infer_target_frames(concept_root: Path, concept_names: list[str]) -> int:
    first_concept_path = concept_npy_paths(concept_root, concept_names[0])[0]
    return int(cast(npt.NDArray[np.generic], np.load(first_concept_path)).shape[1])


def normalize_frames(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Adjust the last dimension of *tensor* to exactly *target_frames*."""
    current_frames = int(tensor.shape[-1])
    if current_frames == target_frames:
        return tensor
    if current_frames > target_frames:
        start = (current_frames - target_frames) // 2
        return tensor[..., start : start + target_frames]

    pad = target_frames - current_frames
    left = pad // 2
    right = pad - left
    return F.pad(tensor, (left, right), mode="constant", value=0.0)


def ensure_random_concept(
    concept_root: Path,
    concept_names: list[str],
    random_concept_name: str,
    random_seed: int,
) -> list[Path]:
    random_dir = concept_dir(concept_root, random_concept_name)
    existing = sorted(random_dir.glob("*.npy")) if random_dir.is_dir() else []
    reference_paths = concept_npy_paths(concept_root, concept_names[0])
    target_count = max(2, len(reference_paths))

    if len(existing) >= target_count:
        return existing

    first_concept_path = reference_paths[0]
    reference_array = cast(npt.NDArray[np.generic], np.load(first_concept_path))
    reference = np.asarray(reference_array, dtype=np.float32)
    rng = np.random.default_rng(random_seed)

    random_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(existing), target_count):
        noise = rng.normal(0.0, 1.0, size=reference.shape).astype(np.float32)
        output_path = random_dir / f"gaussian_noise_{idx:03d}.npy"
        np.save(output_path, noise)

    return sorted(random_dir.glob("*.npy"))


def load_npy_as_tensor(filename: str) -> torch.Tensor:
    """Load a concept .npy stored as (F, T) and return as (T, F) for ResNet293.

    ResNet293 expects input shape (batch, T, F).  Concept arrays are saved
    as (F, T) – the same layout used by the concept-creation code.  We
    normalise frames (last dim while still in F,T layout) then transpose.
    """
    array = cast(npt.NDArray[np.generic], np.load(filename))
    return torch.from_numpy(array).float()  # (F, T) -- 2-D


def make_iter(
    *,
    concept_root: Path,
    concept_names: list[str],
    random_concept_name: str,
    random_seed: int,
    name: str,
):
    from captum.concept._utils.data_iterator import (
        CustomIterableDataset,
        dataset_to_dataloader,
    )

    target_frames = infer_target_frames(concept_root, concept_names)

    def loader(filename: str) -> torch.Tensor:
        tensor = load_npy_as_tensor(filename)       # (F, T)
        tensor = normalize_frames(tensor, target_frames)  # adjust T (last dim)
        tensor = tensor.permute(1, 0)               # (T, F) -- model input format
        return tensor

    if name == random_concept_name:
        _ = ensure_random_concept(
            concept_root, concept_names, random_concept_name, random_seed
        )
    else:
        _ = concept_npy_paths(concept_root, name)
    dataset = CustomIterableDataset(
        loader,
        os.path.join(str(concept_dir(concept_root, name)), ""),
    )
    return dataset_to_dataloader(dataset)
