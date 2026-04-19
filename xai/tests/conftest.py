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


# ========================================================================
# Phase 5 (SHAP baseline) fixtures -- append-only, do NOT modify fixtures
# above this block (they are used by Phase 1-4 tests).
# ========================================================================
import yaml  # noqa: E402 (appended after existing imports)


@pytest.fixture
def synthetic_acoustic_features():
    """Deterministic (X, y) for classifier + SHAP tests.

    Two linearly separable Gaussians in 64-dim feature space; 80% class 0 / 20%
    class 1 to mimic ASVspoof5-like imbalance. Seeded for determinism across
    numpy versions (np.random.RandomState, NOT np.random.seed -- 02-02-SUMMARY pattern).
    """
    rng = np.random.RandomState(42)
    n_features = 64
    n_0, n_1 = 400, 100
    mu_0 = np.zeros(n_features)
    mu_1 = np.zeros(n_features); mu_1[:5] = 2.0
    X_0 = rng.multivariate_normal(mu_0, np.eye(n_features), size=n_0)
    X_1 = rng.multivariate_normal(mu_1, np.eye(n_features), size=n_1)
    X = np.vstack([X_0, X_1])
    y = np.array([0] * n_0 + [1] * n_1)
    names = [f'feat_{i}' for i in range(n_features)]
    return {'X': X, 'y': y, 'feature_names': names}


@pytest.fixture
def fake_shap_config(tmp_path):
    """Minimal shap_config-shaped dict + yaml file for cache tests."""
    cfg = {
        'features': {
            'sample_rate': 16000, 'n_mfcc': 13, 'include_mfcc_deltas': True,
            'f0_min': 75, 'f0_max': 500, 'pause_rate_threshold_ratio': 0.1,
        },
        'classifier': {
            'test_size': 0.2,
            'rf': {'n_estimators': 200, 'class_weight': 'balanced'},
            'svm': {'kernel': 'rbf', 'class_weight': 'balanced', 'probability': True},
            'seed': 42,
        },
        'shap': {'n_background': 10, 'n_eval': 50, 'nsamples': 100, 'target_class': 1},
        'concept_to_features': {
            'breathiness':         ['HNR_mean', 'HNR_var'],
            'pitch_monotony':      ['F0_std', 'F0_mean'],
            'spectral_smoothness': ['spectral_flux_var', 'spectral_centroid_var'],
            'temporal_regularity': ['energy_envelope_var', 'pause_rate'],
        },
        'comparison': {'rho_confirm': 0.7, 'rho_challenge': 0.3},
    }
    cfg_path = tmp_path / 'shap_config.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))
    return {'cfg': cfg, 'path': str(cfg_path)}


@pytest.fixture
def synthetic_tcav_results():
    """Phase-4-shaped per_system_results.json stub (subset of keys) for compare.py tests."""
    return {
        'systems_analyzed':  ['A09', 'A10', 'A11', 'A12'],
        'systems_excluded':  {},
        'per_system_scores': {
            'A09': {'breathiness': {'post_gru': {'mean_score': 0.82, 'significant': True}}},
            'A10': {'breathiness': {'post_gru': {'mean_score': 0.75, 'significant': True}}},
            'A11': {'breathiness': {'post_gru': {'mean_score': 0.55, 'significant': False}}},
            'A12': {'breathiness': {'post_gru': {'mean_score': 0.88, 'significant': True}}},
        },
        'concept_signatures': {
            'A09': {'significant_concepts': ['breathiness']},
            'A10': {'significant_concepts': ['breathiness']},
            'A11': {'significant_concepts': []},
            'A12': {'significant_concepts': ['breathiness']},
        },
        'transferability': {
            'breathiness': {
                'classification': 'universal',
                'n_significant_systems': 3,
                'significant_systems': ['A09', 'A10', 'A12'],
                'best_layer': 'post_gru',
            }
        },
        'n_significant': 3,
        'fdr_method':    'benjamini_hochberg',
        'alpha':         0.05,
        'n_tests_total': 160,
    }


@pytest.fixture
def synthetic_shap_results():
    """Full shap_results.json stub exercising every validate_shap gate (all-pass configuration)."""
    return {
        'classifier': {
            'rf':  {'accuracy': 0.92, 'auc': 0.78, 'confusion_matrix': [[380, 20], [50, 50]]},
            'svm': {'accuracy': 0.88, 'auc': 0.66, 'confusion_matrix': [[360, 40], [60, 40]]},
            'best_classifier': 'rf',
        },
        'feature_importances':  [{'name': f'feat_{i}', 'mean_abs_shap': 0.1 - 0.001 * i}
                                 for i in range(64)],
        'top_15_features':      [[f'feat_{i}', 0.1 - 0.001 * i] for i in range(15)],
        'per_concept_rho':      {'breathiness': {'rho': 0.82, 'pval': 0.01, 'n_mapped': 2}},
        'overall_rho':          {'rho': 0.71, 'pval': 0.04},
        'comparison_status':    'confirms',
        'n_eval_used':          100,
        'n_background':         50,
        'feature_names':        [f'feat_{i}' for i in range(64)],
    }
