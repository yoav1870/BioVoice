"""Inference pipeline for running RawNet2 on ASVspoof data."""

import os
import sys
import tarfile
import io
import time
from typing import Optional

import numpy as np
import torch
import torchaudio
import yaml


def _parse_protocol(protocol_path, partition="dev", max_samples=None):
    """Parse ASVspoof5 protocol TSV file.

    ASVspoof5 dev protocol format (space/tab separated):
    speaker_id audio_id gender ? ? ? codec attack_type key ?

    Args:
        protocol_path: Path to the protocol .tsv file.
        partition: Partition name (for logging).
        max_samples: Maximum number of samples to return, None for all.

    Returns:
        List of (audio_id, label, attack_type) tuples.
        label: 1 = bonafide, 0 = spoof.
    """
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            audio_id = parts[1]
            key = parts[8]  # "bonafide" or "spoof"
            attack_type = parts[7]  # e.g., "A11", "bonafide"
            label = 1 if key == "bonafide" else 0
            entries.append((audio_id, label, attack_type))

    if max_samples is not None and max_samples < len(entries):
        # Stratified subsample: maintain bonafide/spoof ratio
        bonafide = [e for e in entries if e[1] == 1]
        spoof = [e for e in entries if e[1] == 0]
        ratio = len(bonafide) / len(entries)
        n_bonafide = max(1, int(max_samples * ratio))
        n_spoof = max_samples - n_bonafide
        np.random.seed(42)
        bonafide_idx = np.random.choice(
            len(bonafide), min(n_bonafide, len(bonafide)), replace=False
        )
        spoof_idx = np.random.choice(
            len(spoof), min(n_spoof, len(spoof)), replace=False
        )
        entries = [bonafide[i] for i in bonafide_idx] + [spoof[i] for i in spoof_idx]

    return entries


def _load_waveform(audio_bytes, target_sr=16000):
    """Load waveform from bytes buffer.

    Returns:
        1D tensor of audio samples, or None if loading fails.
    """
    try:
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0)
    except Exception as e:
        return None


def _process_batch(model, batch_waveforms, device):
    """Run model on a batch and return bonafide scores."""
    batch_tensor = torch.stack(batch_waveforms).to(device)
    with torch.no_grad():
        output = model(batch_tensor)  # (batch, 2) log-softmax
    # Score = bonafide class log-probability (index 1)
    # In ASVspoof convention: label 1 = bonafide, label 0 = spoof
    # Model output index matches label index: output[:, 1] = bonafide score
    # Higher score = more likely bonafide
    return output[:, 1].cpu().numpy()


def run_inference(model, config, device):
    """Run model on ASVspoof partition and compute scores.

    Efficient implementation: streams through tar files once, keeping each
    tar open and extracting needed files sequentially. This avoids the
    O(N * tar_count) cost of reopening tars per file.

    Args:
        model: RawNet2 model in eval mode.
        config: Experiment config dict (loaded from experiment.yaml).
        device: Compute device string.

    Returns:
        (labels, scores) numpy arrays.
        labels: 1=bonafide, 0=spoof.
        scores: higher = more likely bonafide (bonafide class log-probability).
    """
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]

    protocol_path = data_cfg["protocol_file"]
    tars_dir = data_cfg["dev_tars_dir"]
    sample_rate = data_cfg["sample_rate"]
    clip_samples = data_cfg["clip_samples"]
    batch_size = eval_cfg["batch_size"]
    max_samples = eval_cfg.get("max_samples")

    # Parse protocol
    print("Parsing protocol: {}".format(protocol_path))
    entries = _parse_protocol(protocol_path, max_samples=max_samples)
    n_bonafide = sum(1 for _, l, _ in entries if l == 1)
    n_spoof = sum(1 for _, l, _ in entries if l == 0)
    print("  Samples: {} bonafide, {} spoof, {} total".format(
        n_bonafide, n_spoof, len(entries)
    ))

    # Build lookup: audio_id -> (label, attack_type)
    needed = {}
    for audio_id, label, attack_type in entries:
        needed[audio_id] = (label, attack_type)

    # Collect all audio data by streaming through tars once each
    # Store: audio_id -> (waveform_tensor, label)
    audio_data = {}
    tar_files = sorted(
        f for f in os.listdir(tars_dir) if f.endswith(".tar")
    )

    print("Loading audio from {} tar file(s)...".format(len(tar_files)))
    load_start = time.time()
    loaded_count = 0
    failed_count = 0

    for tar_name in tar_files:
        tar_path = os.path.join(tars_dir, tar_name)
        print("  Reading {}...".format(tar_name))
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".flac"):
                    continue
                basename = os.path.splitext(os.path.basename(member.name))[0]
                if basename not in needed:
                    continue

                # Extract and load audio
                f = tf.extractfile(member)
                if f is None:
                    failed_count += 1
                    continue
                audio_bytes = f.read()
                waveform = _load_waveform(audio_bytes, target_sr=sample_rate)
                if waveform is None:
                    failed_count += 1
                    continue

                # Truncate or pad to clip_samples
                if waveform.shape[0] > clip_samples:
                    waveform = waveform[:clip_samples]
                elif waveform.shape[0] < clip_samples:
                    padding = torch.zeros(clip_samples - waveform.shape[0])
                    waveform = torch.cat([waveform, padding])

                label = needed[basename][0]
                audio_data[basename] = (waveform, label)
                loaded_count += 1

                if loaded_count % 5000 == 0:
                    elapsed = time.time() - load_start
                    print("    Loaded {}/{} files ({:.0f}s elapsed)...".format(
                        loaded_count, len(needed), elapsed
                    ))

    load_elapsed = time.time() - load_start
    print("  Audio loading complete: {} loaded, {} failed ({:.1f}s)".format(
        loaded_count, failed_count, load_elapsed
    ))

    # Run inference in batches (preserving protocol order for reproducibility)
    all_labels = []
    all_scores = []
    batch_waveforms = []
    batch_labels = []
    processed = 0
    skipped = 0

    print("Running inference (batch_size={})...".format(batch_size))
    model.eval()
    infer_start = time.time()

    for audio_id, label, attack_type in entries:
        if audio_id not in audio_data:
            skipped += 1
            continue

        waveform, _ = audio_data[audio_id]
        batch_waveforms.append(waveform)
        batch_labels.append(label)

        if len(batch_waveforms) == batch_size:
            scores = _process_batch(model, batch_waveforms, device)
            all_scores.extend(scores)
            all_labels.extend(batch_labels)
            batch_waveforms = []
            batch_labels = []
            processed += len(scores)

            if processed % (batch_size * 100) == 0:
                elapsed = time.time() - infer_start
                print("  Processed {}/{} ({:.0f}s elapsed)...".format(
                    processed, len(entries), elapsed
                ))

    # Process remaining samples
    if batch_waveforms:
        scores = _process_batch(model, batch_waveforms, device)
        all_scores.extend(scores)
        all_labels.extend(batch_labels)
        processed += len(scores)

    infer_elapsed = time.time() - infer_start
    print("  Inference complete: {} processed, {} skipped ({:.1f}s)".format(
        processed, skipped, infer_elapsed
    ))

    return np.array(all_labels), np.array(all_scores)
