"""Pytest configuration and shared fixtures for xai tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure xai package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def device():
    """Return available compute device."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def xai_root():
    """Return the root path of the xai module."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def model_config_path(xai_root):
    """Return path to model_config_RawNet.yaml."""
    path = xai_root / "config" / "model_config_RawNet.yaml"
    if not path.exists():
        pytest.skip(f"Model config not found: {path}")
    return str(path)


@pytest.fixture
def weights_path(xai_root):
    """Return path to pretrained weights file. Skip test if not found."""
    path = xai_root / "models" / "weights" / "pre_trained_DF_RawNet2.pth"
    if not path.exists():
        pytest.skip(f"Pretrained weights not found: {path}")
    return str(path)


@pytest.fixture
def sample_audio():
    """Return a batch of random audio tensors matching expected input shape."""
    return torch.randn(2, 64600)


@pytest.fixture
def concept_config_path(xai_root):
    """Return path to concepts.yaml config file."""
    return os.path.join(xai_root, "config", "concepts.yaml")


@pytest.fixture
def synthetic_audio():
    """1-second 440Hz sine wave at 16kHz, float64."""
    sr = 16000
    t = np.arange(sr) / sr
    audio = np.sin(2 * np.pi * 440 * t)
    return audio, sr


@pytest.fixture
def mock_features_csv(tmp_path):
    """Create a mock bonafide_features.csv with 20 rows for testing filtering."""
    import csv as csv_mod
    csv_path = str(tmp_path / "bonafide_features.csv")
    rows = []
    for i in range(20):
        speaker = f"T_{i // 4:04d}"  # 5 speakers, 4 clips each
        rows.append({
            "audio_id": f"T_{i:010d}",
            "speaker_id": speaker,
            "mean_hnr": str(5.0 + i * 0.8),
            "f0_std": str(15.0 + i * 3.0),
            "spectral_flux_var": str(0.005 + i * 0.003),
            "energy_envelope_var": str(0.0002 + i * 0.0002),
        })
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(
            f,
            fieldnames=["audio_id", "speaker_id", "mean_hnr", "f0_std",
                        "spectral_flux_var", "energy_envelope_var"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


@pytest.fixture
def fake_protocol_file(tmp_path):
    """Create a minimal ASVspoof5-format protocol file for testing."""
    protocol = tmp_path / 'dev_protocol.tsv'
    # 9 columns, space-delimited (NOT tab -- Phase 3 bug fix 7e42e7c)
    # col[0]=speaker, col[1]=audio_id, col[7]=attack_type, col[8]=key
    lines = [
        'SPK001 D_00001 - - - - - A09 spoof',
        'SPK001 D_00002 - - - - - A09 spoof',
        'SPK002 D_00003 - - - - - A10 spoof',
        'SPK002 D_00004 - - - - - A10 spoof',
        'SPK003 D_00005 - - - - - A10 spoof',
        'SPK003 D_00006 - - - - - - bonafide',
        'SPK004 D_00007 - - - - - - bonafide',
        'SPK004 D_00008 - - - - - A12 spoof',
    ]
    protocol.write_text('\n'.join(lines) + '\n')
    return str(protocol)


@pytest.fixture
def synthetic_per_system_scores():
    """Create synthetic per-system TCAV score matrices for testing."""
    systems = ['A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
    concepts = ['breathiness', 'pitch_monotony', 'spectral_smoothness', 'temporal_regularity', 'negative_control']
    layers = ['sinc_conv', 'resblock_g1', 'resblock_g2', 'pre_gru', 'post_gru']
    rng = np.random.RandomState(42)
    scores = {}
    for sys_id in systems:
        scores[sys_id] = {}
        for concept in concepts:
            scores[sys_id][concept] = {}
            for layer in layers:
                scores[sys_id][concept][layer] = {
                    'tcav_scores': [float(x) for x in rng.uniform(0.3, 0.9, size=10)],
                    'mean_score': float(rng.uniform(0.3, 0.9)),
                    'pval': float(rng.uniform(0.001, 0.5)),
                }
    return {'systems': systems, 'concepts': concepts, 'layers': layers, 'scores': scores}
