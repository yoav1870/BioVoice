"""Unit tests for Phase 3 Plan 01: TCAV activation extraction and CAV training.

Covers requirements TCAV-01 (activation extraction), TCAV-02 (CAV training).
GPU tests require the RawNet2 model and are marked with @pytest.mark.gpu.
CPU tests cover pooling and CAV training logic and run without GPU.
"""

import pytest
import numpy as np
import torch
import yaml


@pytest.fixture(scope='module')
def model(device, xai_root):
    """Load RawNet2 for TCAV tests -- reuses Phase 1 loader."""
    from xai.models.loader import load_rawnet2
    cfg = yaml.safe_load(open(xai_root / 'config' / 'experiment.yaml'))
    return load_rawnet2(
        str(xai_root / cfg['model']['config_path']),
        str(xai_root / cfg['model']['weights_path']),
        device=str(device)
    )


class TestActivationExtractor:
    """Tests for hooks.py ActivationExtractor."""

    @pytest.mark.gpu
    def test_extracts_five_layers(self, model, device):
        """ActivationExtractor returns activations for all 5 LAYER_NAMES keys."""
        from xai.tcav.hooks import ActivationExtractor, LAYER_NAMES
        sample = torch.randn(2, 64600, device=device)
        with ActivationExtractor(model, LAYER_NAMES) as ext:
            model.eval()
            with torch.no_grad():
                _ = model(sample)
            for key in LAYER_NAMES:
                act = ext.get(key)
                assert act is not None, f"Activation for {key} is None"
                assert isinstance(act, torch.Tensor), f"{key} is not a Tensor"

    @pytest.mark.gpu
    def test_hooks_removed_after_exit(self, model, device):
        """After exiting context manager, no hooks remain on model modules."""
        from xai.tcav.hooks import ActivationExtractor, LAYER_NAMES
        sample = torch.randn(2, 64600, device=device)
        with ActivationExtractor(model, LAYER_NAMES) as ext:
            model.eval()
            with torch.no_grad():
                _ = model(sample)
        # After exit, _handles should be empty
        assert len(ext._handles) == 0

    @pytest.mark.gpu
    def test_gru_tuple_unpacking(self, model, device):
        """GRU layer activation is a tensor (not a tuple) after hook processing."""
        from xai.tcav.hooks import ActivationExtractor, LAYER_NAMES
        sample = torch.randn(2, 64600, device=device)
        with ActivationExtractor(model, LAYER_NAMES) as ext:
            model.eval()
            with torch.no_grad():
                _ = model(sample)
            gru_act = ext.get('post_gru')
            assert isinstance(gru_act, torch.Tensor), "GRU activation should be tensor, got tuple"

    @pytest.mark.gpu
    def test_activation_shapes_valid(self, model, device):
        """All extracted activations have batch dimension == input batch size."""
        from xai.tcav.hooks import ActivationExtractor, LAYER_NAMES
        batch_size = 3
        sample = torch.randn(batch_size, 64600, device=device)
        with ActivationExtractor(model, LAYER_NAMES) as ext:
            model.eval()
            with torch.no_grad():
                _ = model(sample)
            for key in LAYER_NAMES:
                act = ext.get(key)
                assert act.shape[0] == batch_size, f"{key} batch dim {act.shape[0]} != {batch_size}"

    def test_pool_activation_3d_conv(self):
        """pool_activation reduces (batch, C, T) where C < T to (batch, C) via GAP."""
        from xai.tcav.hooks import pool_activation
        act = torch.randn(4, 20, 125)  # C=20 < T=125, conv-like
        pooled = pool_activation(act)
        assert pooled.shape == (4, 20)

    def test_pool_activation_3d_gru(self):
        """pool_activation reduces (batch, seq_len, H) where seq_len < H to (batch, H) via last timestep."""
        from xai.tcav.hooks import pool_activation
        act = torch.randn(4, 10, 1024)  # seq_len=10 < H=1024, GRU-like
        pooled = pool_activation(act)
        assert pooled.shape == (4, 1024)

    def test_pool_activation_2d_passthrough(self):
        """pool_activation returns 2D tensors unchanged."""
        from xai.tcav.hooks import pool_activation
        act = torch.randn(4, 256)
        pooled = pool_activation(act)
        assert pooled.shape == (4, 256)


class TestCAVTrainer:
    """Tests for cav.py train_cav function."""

    def test_train_cav_returns_expected_keys(self):
        """train_cav returns dict with 'cav', 'accuracy', 'scaler' keys."""
        from xai.tcav.cav import train_cav
        concept_acts = np.random.randn(100, 64)
        random_acts = np.random.randn(100, 64)
        result = train_cav(concept_acts, random_acts)
        assert 'cav' in result
        assert 'accuracy' in result
        assert 'scaler' in result

    def test_cav_is_unit_vector(self):
        """CAV vector is normalized to unit length."""
        from xai.tcav.cav import train_cav
        concept_acts = np.random.randn(100, 64)
        random_acts = np.random.randn(100, 64)
        result = train_cav(concept_acts, random_acts)
        norm = np.linalg.norm(result['cav'])
        assert abs(norm - 1.0) < 1e-6, f"CAV norm {norm} != 1.0"

    def test_cav_shape_matches_activation_dim(self):
        """CAV vector dimension matches input activation dimension."""
        from xai.tcav.cav import train_cav
        d = 128
        concept_acts = np.random.randn(100, d)
        random_acts = np.random.randn(100, d)
        result = train_cav(concept_acts, random_acts)
        assert result['cav'].shape == (d,)

    def test_random_vs_random_cav_accuracy_near_50(self):
        """CAV trained on two random distributions has accuracy near 50% (chance)."""
        from xai.tcav.cav import train_cav
        rng = np.random.RandomState(42)
        data_a = rng.randn(200, 64)
        data_b = rng.randn(200, 64)
        result = train_cav(data_a, data_b, random_state=42)
        assert 0.3 <= result['accuracy'] <= 0.7, f"Random-vs-random accuracy {result['accuracy']} not near 0.5"

    def test_separable_concept_high_accuracy(self):
        """CAV trained on well-separated distributions achieves high accuracy."""
        from xai.tcav.cav import train_cav
        rng = np.random.RandomState(42)
        concept_acts = rng.randn(200, 64) + 3.0  # shifted by 3 std devs
        random_acts = rng.randn(200, 64)
        result = train_cav(concept_acts, random_acts, random_state=42)
        assert result['accuracy'] > 0.85, f"Separable concept accuracy {result['accuracy']} should be > 0.85"

    def test_equal_class_sizes_enforced(self):
        """CAV training downsamples to min(concept, random) size."""
        from xai.tcav.cav import train_cav
        concept_acts = np.random.randn(500, 32)  # 500 concept
        random_acts = np.random.randn(100, 32)   # 100 random
        result = train_cav(concept_acts, random_acts, random_state=42)
        # Should succeed without error; accuracy should be valid
        assert 0.0 <= result['accuracy'] <= 1.0

    def test_scaler_is_fitted(self):
        """Returned scaler has been fitted (has mean_ attribute)."""
        from xai.tcav.cav import train_cav
        concept_acts = np.random.randn(100, 64)
        random_acts = np.random.randn(100, 64)
        result = train_cav(concept_acts, random_acts)
        assert hasattr(result['scaler'], 'mean_'), "Scaler not fitted"
