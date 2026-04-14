"""Pytest configuration and shared fixtures for xai tests."""

import os
import sys
from pathlib import Path

import pytest
import torch

# Ensure xai package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def device():
    """Return available compute device."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
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
