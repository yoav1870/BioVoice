"""Concept audio loading utilities for TCAV pipeline.

Loads concept audio clips from Phase 2 manifests and samples random baselines
from the bonafide features pool. All sampling uses seeded RandomState for
reproducibility, and tar-streaming with path traversal prevention.
"""

import csv
import io
import os
import tarfile
from pathlib import Path
from typing import Optional, Set

import numpy as np
import soundfile as sf
import torch


def load_concept_clips(
    manifest_path: str,
    tar_dir: str,
    n_clips: int = 200,
    seed: int = 42,
    fixed_len: int = 64000,
) -> torch.Tensor:
    """Load concept audio clips from a Phase 2 manifest CSV using tar-streaming.

    Uses seeded deterministic sampling (np.random.RandomState -- NOT np.random.seed)
    and sorts selected clips by audio_id for reproducibility.

    Security: os.path.basename(member.name) prevents tar path traversal (T-03-01).

    Args:
        manifest_path: Path to concept manifest.csv file.
        tar_dir: Directory containing *.tar audio archives.
        n_clips: Number of clips to load (sampled from manifest).
        seed: Random seed for deterministic sampling.
        fixed_len: Fixed waveform length in samples (zero-pad or truncate).

    Returns:
        Tensor of shape (n_clips, 1, fixed_len) dtype float32.
    """
    # Read manifest and sample rows
    with open(manifest_path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))

    # Use RandomState for deterministic sampling (Phase 2 pattern)
    rng = np.random.RandomState(seed)
    n_select = min(n_clips, len(rows))
    selected_indices = rng.choice(len(rows), n_select, replace=False)
    selected_rows = [rows[i] for i in selected_indices]

    # Sort by audio_id for reproducibility (Phase 2 pattern)
    selected_rows = sorted(selected_rows, key=lambda r: r['audio_id'])
    needed_set = {r['audio_id'] for r in selected_rows}

    # Stream through tar files to collect waveforms
    waveforms: dict = {}
    for tar_path in sorted(Path(tar_dir).glob('*.tar')):
        if len(waveforms) >= n_select:
            break
        try:
            with tarfile.open(str(tar_path), 'r') as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    # Prevent path traversal (T-03-01 mitigation)
                    basename = os.path.basename(member.name)
                    # Strip .flac extension to match audio_id format
                    audio_id = basename[:-5] if basename.endswith('.flac') else basename
                    if audio_id not in needed_set:
                        continue
                    file_obj = tar.extractfile(member)
                    if file_obj is None:
                        continue
                    audio_bytes = file_obj.read()
                    wav, sr = sf.read(io.BytesIO(audio_bytes))
                    # Convert to mono float32 if needed
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    wav = wav.astype(np.float32)
                    # Zero-pad or truncate to fixed_len
                    if len(wav) < fixed_len:
                        wav = np.pad(wav, (0, fixed_len - len(wav)))
                    else:
                        wav = wav[:fixed_len]
                    waveforms[audio_id] = wav
                    if len(waveforms) >= n_select:
                        break
        except (tarfile.TarError, OSError):
            continue

    # Assemble in sorted order
    clips = []
    for row in selected_rows:
        aid = row['audio_id']
        if aid in waveforms:
            clips.append(waveforms[aid])

    if not clips:
        raise RuntimeError(
            f"No audio clips found for manifest {manifest_path} in tar_dir {tar_dir}. "
            f"Searched for {len(needed_set)} audio IDs."
        )

    tensor = torch.tensor(np.stack(clips, axis=0), dtype=torch.float32)  # (n, fixed_len)
    return tensor.unsqueeze(1)  # (n, 1, fixed_len)


def sample_random_baseline(
    manifest_dir: str,
    tar_dir: str,
    n_clips: int = 200,
    seed: int = 0,
    exclude_audio_ids: Optional[Set[str]] = None,
    fixed_len: int = 64000,
) -> torch.Tensor:
    """Sample random bonafide baseline clips from the features CSV.

    Loads all bonafide audio IDs from bonafide_features.csv, excludes any IDs
    in exclude_audio_ids (concept clips + other baselines), then samples n_clips
    with seeded RandomState.

    Args:
        manifest_dir: Directory containing xai/data/features/bonafide_features.csv
                      (relative to this directory) or absolute path to bonafide_features.csv.
        tar_dir: Directory containing *.tar audio archives.
        n_clips: Number of random clips to sample.
        seed: Random seed (each baseline uses a different seed = baseline_index).
        exclude_audio_ids: Set of audio_ids to exclude (concept clips + prior baselines).
        fixed_len: Fixed waveform length in samples (zero-pad or truncate).

    Returns:
        Tensor of shape (n_clips, 1, fixed_len) dtype float32.
    """
    if exclude_audio_ids is None:
        exclude_audio_ids = set()

    # Locate bonafide_features.csv: try direct path first, then relative subpath
    features_path = Path(manifest_dir)
    if features_path.is_file():
        csv_path = features_path
    else:
        csv_path = features_path / 'data' / 'features' / 'bonafide_features.csv'
        if not csv_path.exists():
            csv_path = features_path / 'bonafide_features.csv'

    with open(str(csv_path), 'r', newline='') as f:
        rows = list(csv.DictReader(f))

    # Filter out excluded IDs
    rows = [r for r in rows if r['audio_id'] not in exclude_audio_ids]

    if len(rows) == 0:
        raise RuntimeError(
            f"No rows remaining in bonafide_features.csv after excluding "
            f"{len(exclude_audio_ids)} IDs."
        )

    # Seeded sampling (different seed per baseline index)
    rng = np.random.RandomState(seed)
    n_select = min(n_clips, len(rows))
    selected_indices = rng.choice(len(rows), n_select, replace=False)
    selected_rows = [rows[i] for i in selected_indices]

    # Sort by audio_id for reproducibility
    selected_rows = sorted(selected_rows, key=lambda r: r['audio_id'])
    needed_set = {r['audio_id'] for r in selected_rows}

    # Stream through tar files
    waveforms: dict = {}
    for tar_path in sorted(Path(tar_dir).glob('*.tar')):
        if len(waveforms) >= n_select:
            break
        try:
            with tarfile.open(str(tar_path), 'r') as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    # Prevent path traversal (T-03-01 mitigation)
                    basename = os.path.basename(member.name)
                    audio_id = basename[:-5] if basename.endswith('.flac') else basename
                    if audio_id not in needed_set:
                        continue
                    file_obj = tar.extractfile(member)
                    if file_obj is None:
                        continue
                    audio_bytes = file_obj.read()
                    wav, sr = sf.read(io.BytesIO(audio_bytes))
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    wav = wav.astype(np.float32)
                    if len(wav) < fixed_len:
                        wav = np.pad(wav, (0, fixed_len - len(wav)))
                    else:
                        wav = wav[:fixed_len]
                    waveforms[audio_id] = wav
                    if len(waveforms) >= n_select:
                        break
        except (tarfile.TarError, OSError):
            continue

    # Assemble in sorted order
    clips = []
    for row in selected_rows:
        aid = row['audio_id']
        if aid in waveforms:
            clips.append(waveforms[aid])

    if not clips:
        raise RuntimeError(
            f"No audio clips found in tar_dir {tar_dir} for random baseline. "
            f"Searched for {len(needed_set)} audio IDs."
        )

    tensor = torch.tensor(np.stack(clips, axis=0), dtype=torch.float32)
    return tensor.unsqueeze(1)  # (n, 1, fixed_len)
