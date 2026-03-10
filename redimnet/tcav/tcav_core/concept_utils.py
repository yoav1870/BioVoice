"""Concept building and management utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from captum.concept import Concept
from torch.utils.data import DataLoader

from tcav_core.config import ExperimentConfig
from tcav_core.datasets import (
    ConceptNPYDataset,
    RandomGaussianMelDataset,
    ShuffledRealMelDataset,
)
from tcav_core.device_utils import DeviceDataLoader
from tcav_core.frame import FrameNormalizer


def list_positive_concept_dirs(
    concept_root: Path, excluded_names: tuple[str, ...]
) -> list[Path]:
    """List concept directories excluding specified names."""
    excluded = {name.strip().lower() for name in excluded_names}
    all_dirs = sorted([d for d in concept_root.iterdir() if d.is_dir()])

    filtered: list[Path] = []
    skipped: list[str] = []
    for concept_dir in all_dirs:
        key = concept_dir.name.strip().lower()
        if key in excluded or key.startswith("random"):
            skipped.append(concept_dir.name)
            continue
        filtered.append(concept_dir)

    if skipped:
        print(f"Skipping concept dirs: {skipped}")
    if not filtered:
        raise RuntimeError(
            f"No positive concept directories found after filtering. "
            f"concept_root={concept_root} excluded={sorted(excluded)}"
        )
    return filtered


def build_concept_datasets(
    concept_dirs: list[Path],
    n_mels: int,
    frame_normalizer: FrameNormalizer,
    config: ExperimentConfig,
    loader_kwargs: dict[str, Any],
    tcav_device: torch.device,
) -> list[Concept]:
    """Build concept datasets and wrap in DeviceDataLoader."""
    positive_concepts: list[Concept] = []

    for idx, concept_dir in enumerate(concept_dirs):
        dataset = ConceptNPYDataset(
            concept_dir=concept_dir,
            n_mels=n_mels,
            frame_normalizer=frame_normalizer,
            seed=config.seed,
            limit=config.concept_samples,
            augment_on_oversample=config.augment_on_oversample,
            augment_time_shift_max=config.concept_augment_time_shift_max,
            augment_freq_shift_max=config.concept_augment_freq_shift_max,
            augment_gain_min=config.concept_augment_gain_min,
            augment_gain_max=config.concept_augment_gain_max,
            augment_noise_std=config.concept_augment_noise_std,
        )

        # if dataset.base_count < config.min_concept_files_warning:
        #     print(
        #         f"WARNING: low concept sample count for '{concept_dir.name}': "
        #         f"{dataset.base_count} (threshold={config.min_concept_files_warning})"
        #     )

        # if (
        #     dataset.requested_count > dataset.base_count
        #     and not config.augment_on_oversample
        # ):
        #     print(
        #         f"WARNING: concept_samples > available files but augment_on_oversample="
        #         f"disabled. '{concept_dir.name}' uses {dataset.base_count} samples."
        #     )

        # print(
        #     f"Concept '{concept_dir.name}': base={dataset.base_count} "
        #     f"requested={dataset.requested_count} effective={len(dataset)} "
        #     f"oversampled={dataset.is_oversampled}"
        # )

        data_loader = DataLoader(dataset, **loader_kwargs)
        data_iter = DeviceDataLoader(data_loader, tcav_device)
        positive_concepts.append(
            Concept(id=idx, name=concept_dir.name, data_iter=data_iter)
        )

    return positive_concepts


def build_random_concepts(
    config: ExperimentConfig,
    n_mels: int,
    target_frames: int,
    frame_normalizer: FrameNormalizer,
    spec_fn: Any,
    random_source_csv: Path,
    loader_kwargs: dict[str, Any],
    tcav_device: torch.device,
) -> list[Concept]:
    """Build random/negative concept datasets."""
    random_concepts: list[Concept] = []
    repeats = max(1, int(config.random_concept_repeats))
    negative_mode = config.negative_mode.strip().lower()

    wav_paths: list[Path] = []
    if negative_mode == "shuffled_real":
        print("Negative concept: shuffled_real")
        random_source_df = pd.read_csv(random_source_csv)
        if config.path_column not in random_source_df.columns:
            raise RuntimeError(
                f"Random source csv must contain '{config.path_column}'. "
                f"Got: {list(random_source_df.columns)}"
            )

        wav_paths = [
            Path(path)
            for path in random_source_df[config.path_column]
            .dropna()
            .astype(str)
            .tolist()
        ]
        wav_paths = [path for path in wav_paths if path.exists()]
        if config.random_source_max_files is not None:
            wav_paths = wav_paths[: int(config.random_source_max_files)]
        if not wav_paths:
            raise RuntimeError(
                "No valid wav files found for shuffled_real negative concept."
            )

        print(
            f"shuffled_real pool={len(wav_paths)} | shuffle_time={config.shuffle_time} | "
            f"shuffle_freq={config.shuffle_freq} | random_repeats={repeats}"
        )
    elif negative_mode == "gaussian":
        print(f"Negative concept: gaussian | random_repeats={repeats}")
    else:
        raise ValueError("negative_mode must be 'gaussian' or 'shuffled_real'.")

    base_id = 1000  # Start random concepts at high IDs to avoid collision

    # Adjust loader kwargs for random concepts if needed
    random_loader_kwargs = dict(loader_kwargs)
    if (
        tcav_device.type == "cuda"
        and int(random_loader_kwargs.get("num_workers", 0)) > 0
        and negative_mode == "shuffled_real"
    ):
        print(
            "INFO: forcing random concept DataLoader num_workers=0 for shuffled_real "
            "to avoid CUDA-in-worker issues."
        )
        random_loader_kwargs["num_workers"] = 0
        random_loader_kwargs["pin_memory"] = False
        random_loader_kwargs.pop("persistent_workers", None)

    for ridx in range(repeats):
        random_seed = int(config.seed + (1000 * ridx))

        if negative_mode == "gaussian":
            random_dataset = RandomGaussianMelDataset(
                n_samples=config.random_samples,
                n_mels=n_mels,
                target_frames=target_frames,
                seed=random_seed,
            )
        else:
            random_dataset = ShuffledRealMelDataset(
                wav_paths=wav_paths,
                n_samples=config.random_samples,
                n_mels=n_mels,
                frame_normalizer=frame_normalizer,
                spec_fn=spec_fn,
                spec_device=tcav_device,
                shuffle_time=config.shuffle_time,
                shuffle_freq=config.shuffle_freq,
                seed=random_seed,
            )

        random_data_loader = DataLoader(random_dataset, **random_loader_kwargs)
        random_data_iter = DeviceDataLoader(random_data_loader, tcav_device)
        random_concepts.append(
            Concept(
                id=base_id + ridx,
                name=f"random_{ridx}",
                data_iter=random_data_iter,
            )
        )

    return random_concepts
