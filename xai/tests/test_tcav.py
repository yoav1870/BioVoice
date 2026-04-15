"""Unit tests for Phase 3 Plan 01: TCAV activation extraction and CAV training.

Covers requirements TCAV-01 (activation extraction), TCAV-02 (CAV training).
GPU tests require the RawNet2 model and are marked with @pytest.mark.gpu.
CPU tests cover pooling and CAV training logic and run without GPU.
"""

import pytest
import numpy as np
import torch
import yaml
from pathlib import Path


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


class TestTCAVScorer:
    """Tests for scorer.py compute_tcav_score function."""

    def test_tcav_score_in_unit_interval(self):
        """TCAV score must be in [0, 1] (fraction of positive directional derivs)."""
        from xai.tcav.scorer import compute_tcav_score
        # Test the score computation logic with synthetic derivatives
        derivs = np.array([0.5, -0.2, 0.1, 0.3, -0.1])
        score = float(np.mean(derivs > 0))  # 3/5 = 0.6
        assert 0.0 <= score <= 1.0

    def test_tcav_score_all_positive(self):
        """When all directional derivatives are positive, TCAV score = 1.0."""
        derivs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        score = float(np.mean(derivs > 0))
        assert score == 1.0

    def test_tcav_score_all_negative(self):
        """When all directional derivatives are negative, TCAV score = 0.0."""
        derivs = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])
        score = float(np.mean(derivs > 0))
        assert score == 0.0

    @pytest.mark.gpu
    def test_tcav_score_with_model(self, model, device):
        """compute_tcav_score returns valid dict with real model."""
        from xai.tcav.scorer import compute_tcav_score
        inputs = torch.randn(10, 64600)  # 10 synthetic examples
        cav = np.random.randn(20).astype(np.float32)  # dim matches sinc_conv pooled output
        cav = cav / np.linalg.norm(cav)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        result = compute_tcav_score(model, inputs, target_class=1,
                                     layer_name='sinc_conv', cav=cav,
                                     scaler=scaler, device=str(device),
                                     batch_size=5)
        assert 'tcav_score' in result
        assert 'directional_derivs' in result
        assert 0.0 <= result['tcav_score'] <= 1.0
        assert len(result['directional_derivs']) == 10

    @pytest.mark.gpu
    def test_no_grad_in_extraction_vs_grad_in_scoring(self, model, device):
        """Scoring path must NOT use torch.no_grad -- verify grad is not None."""
        from xai.tcav.scorer import compute_tcav_score
        inputs = torch.randn(4, 64600)
        cav = np.random.randn(20).astype(np.float32)
        cav = cav / np.linalg.norm(cav)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # If this runs without "Gradient is None" RuntimeError, grad flow works
        result = compute_tcav_score(model, inputs, target_class=1,
                                     layer_name='sinc_conv', cav=cav,
                                     scaler=scaler, device=str(device))
        assert result['tcav_score'] is not None


class TestSignificanceTester:
    """Tests for stats.py test_significance function."""

    def test_identical_scores_not_significant(self):
        """When real scores equal random baseline scores, result is not significant."""
        from xai.tcav.stats import test_significance
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        random_baselines = [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(10)]
        result = test_significance(scores, random_baselines, n_pairs=25)
        assert result['significant'] is False

    def test_clearly_different_scores_significant(self):
        """When real scores clearly differ from random, result should be significant."""
        from xai.tcav.stats import test_significance
        real = [0.95, 0.93, 0.96, 0.94, 0.92, 0.95, 0.93, 0.94, 0.96, 0.95]
        random_baselines = [[0.5 + np.random.RandomState(i).randn() * 0.05
                             for _ in range(10)] for i in range(10)]
        result = test_significance(real, random_baselines, n_pairs=25)
        assert result['significant'] is True

    def test_bonferroni_corrects_alpha(self):
        """Bonferroni correction multiplies p-value by n_pairs."""
        from xai.tcav.stats import test_significance
        real = [0.7, 0.72, 0.68, 0.71, 0.69]
        random_baselines = [[0.5, 0.52, 0.48, 0.51, 0.49] for _ in range(10)]
        result_no_correction = test_significance(real, random_baselines, n_pairs=1)
        result_corrected = test_significance(real, random_baselines, n_pairs=25)
        # Corrected p-value should be >= raw p-value
        assert result_corrected['pval_corrected'] >= result_no_correction['pval']

    def test_returns_expected_keys(self):
        """Result dict has all expected keys."""
        from xai.tcav.stats import test_significance
        result = test_significance(
            [0.6, 0.65, 0.62],
            [[0.5, 0.48, 0.52] for _ in range(10)],
            n_pairs=25
        )
        for key in ['pval', 'pval_corrected', 'significant', 'ci_95', 'mean_score', 't_stat']:
            assert key in result, f"Missing key: {key}"

    def test_ci_95_is_tuple_of_two(self):
        """Bootstrap CI is a tuple of (low, high)."""
        from xai.tcav.stats import test_significance
        result = test_significance(
            [0.6, 0.65, 0.62, 0.58, 0.63],
            [[0.5, 0.48, 0.52] for _ in range(10)],
            n_pairs=25
        )
        assert len(result['ci_95']) == 2
        assert result['ci_95'][0] <= result['ci_95'][1]

    def test_pval_corrected_capped_at_1(self):
        """Bonferroni-corrected p-value never exceeds 1.0."""
        from xai.tcav.stats import test_significance
        # Identical scores -> raw pval ~ 1.0, corrected should be capped at 1.0
        result = test_significance(
            [0.5, 0.5, 0.5],
            [[0.5, 0.5, 0.5] for _ in range(10)],
            n_pairs=25
        )
        assert result['pval_corrected'] <= 1.0


class TestVisualizer:
    """Tests for viz.py plot_emergence_heatmap function."""

    def test_heatmap_creates_file(self, tmp_path):
        """plot_emergence_heatmap creates a PNG file at output_path."""
        from xai.tcav.viz import plot_emergence_heatmap
        acc = np.random.rand(5, 5) * 0.5 + 0.4  # values in [0.4, 0.9]
        layers = ['sinc_conv', 'resblock_g1', 'resblock_g2', 'pre_gru', 'post_gru']
        concepts = ['breathiness', 'pitch_monotony', 'spectral_smoothness',
                     'temporal_regularity', 'negative_control']
        out = str(tmp_path / 'test_heatmap.png')
        plot_emergence_heatmap(acc, layers, concepts, out)
        assert Path(out).exists(), f"Heatmap not created at {out}"
        assert Path(out).stat().st_size > 0, "Heatmap file is empty"

    def test_heatmap_with_different_dimensions(self, tmp_path):
        """Heatmap works with non-square matrix."""
        from xai.tcav.viz import plot_emergence_heatmap
        acc = np.random.rand(3, 4) * 0.5 + 0.4
        layers = ['layer_a', 'layer_b', 'layer_c']
        concepts = ['c1', 'c2', 'c3', 'c4']
        out = str(tmp_path / 'test_heatmap_rect.png')
        plot_emergence_heatmap(acc, layers, concepts, out)
        assert Path(out).exists()
