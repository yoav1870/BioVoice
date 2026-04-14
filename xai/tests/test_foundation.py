"""Foundation tests for Phase 1: Model loading, EER computation, and module integrity.

Covers requirements FOUND-01 (model loads, EER matches baseline) and FOUND-04 (self-contained module).
"""

import os
import glob
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


class TestModelLoading:
    """Tests for RawNet2 model loading (FOUND-01)."""

    def test_model_loads_without_error(self, model_config_path, weights_path, device):
        """Load model via loader -- assert no exception."""
        from models.loader import load_rawnet2

        model = load_rawnet2(model_config_path, weights_path, device=device, verbose=False)
        assert model is not None
        assert not model.training, "Model should be in eval mode"

    def test_model_output_shape(self, model_config_path, weights_path, device, sample_audio):
        """Assert output shape is (batch, 2)."""
        from models.loader import load_rawnet2

        model = load_rawnet2(model_config_path, weights_path, device=device, verbose=False)
        x = sample_audio.to(device)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 2), "Expected (2, 2), got {}".format(output.shape)

    def test_model_has_expected_layers(self, model_config_path, weights_path, device):
        """Verify model contains expected named layers."""
        from models.loader import load_rawnet2

        model = load_rawnet2(model_config_path, weights_path, device=device, verbose=False)
        module_names = {name for name, _ in model.named_modules()}
        expected = {
            "Sinc_conv", "first_bn",
            "block0", "block1", "block2", "block3", "block4", "block5",
            "bn_before_gru", "gru", "fc1_gru", "fc2_gru",
        }
        missing = expected - module_names
        assert not missing, "Missing expected layers: {}".format(missing)


class TestEERComputation:
    """Tests for EER computation."""

    def test_eer_computation_known_values(self):
        """Test compute_eer with synthetic known labels/scores where EER is analytically known."""
        from evaluation.eer import compute_eer

        # Perfect separation: EER should be ~0
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.85, 0.95, 0.1, 0.2, 0.15, 0.05])
        eer, threshold = compute_eer(labels, scores)
        assert eer < 0.05, "EER should be near 0 for perfect separation, got {}".format(eer)

    def test_eer_computation_overlapping(self):
        """Test compute_eer with overlapping scores."""
        from evaluation.eer import compute_eer

        # Overlapping scores: EER should be non-trivial
        np.random.seed(42)
        n = 1000
        labels = np.concatenate([np.ones(n), np.zeros(n)])
        scores = np.concatenate([
            np.random.normal(0.6, 0.2, n),  # bonafide
            np.random.normal(0.4, 0.2, n),  # spoof
        ])
        eer, threshold = compute_eer(labels, scores)
        assert 0.1 < eer < 0.4, "EER should be moderate for overlapping, got {}".format(eer)

    def test_eer_input_validation(self):
        """Test that compute_eer raises on invalid inputs."""
        from evaluation.eer import compute_eer

        with pytest.raises(ValueError):
            compute_eer(np.array([1, 0]), np.array([0.5]))  # mismatched shape
        with pytest.raises(ValueError):
            compute_eer(np.array([]), np.array([]))  # empty
        with pytest.raises(ValueError):
            compute_eer(np.array([1, 1]), np.array([0.5, 0.5]))  # single class


class TestModuleIntegrity:
    """Tests for xai/ module self-containment (FOUND-04)."""

    def test_xai_no_external_imports(self, xai_root):
        """Check that no .py file in xai/ imports from outside xai/ or standard library."""
        # Known allowed import prefixes
        allowed_prefixes = {
            # Standard library
            "os", "sys", "re", "glob", "pathlib", "collections", "datetime",
            "subprocess", "random", "io", "tarfile", "abc", "typing", "math",
            "functools", "itertools", "contextlib", "warnings", "json", "csv",
            "copy", "time", "argparse", "logging", "hashlib", "tempfile",
            # Third-party (in venv)
            "torch", "torchaudio", "numpy", "np", "scipy", "sklearn",
            "yaml", "matplotlib", "soundfile", "sf", "pytest",
            # Internal xai imports (relative or absolute within xai)
            "models", "evaluation", "config", "experiments", "scripts",
            ".models", ".evaluation", ".config", ".experiments", ".scripts",
        }

        py_files = glob.glob(str(xai_root / "**" / "*.py"), recursive=True)
        violations = []
        import_re = re.compile(
            r"^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))"
        )
        for fpath in py_files:
            with open(fpath, "r") as f:
                for lineno, line in enumerate(f, 1):
                    m = import_re.match(line)
                    if not m:
                        continue
                    module = m.group(1) or m.group(2)
                    top_level = module.split(".")[0]
                    if module.startswith("."):
                        continue  # relative import, OK
                    if top_level in allowed_prefixes:
                        continue
                    violations.append("{}:{}: {}".format(fpath, lineno, line.strip()))

        assert not violations, (
            "Found imports from outside xai/ or allowed packages:\n"
            + "\n".join(violations)
        )


class TestConfig:
    """Tests for configuration files."""

    def test_config_yaml_valid(self, xai_root):
        """Load experiment.yaml with safe_load, assert required keys exist."""
        config_path = xai_root / "config" / "experiment.yaml"
        assert config_path.exists(), "Config not found: {}".format(config_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert "experiment" in config, "Missing experiment key"
        assert "model" in config, "Missing model key"
        assert "data" in config, "Missing data key"
        assert "evaluation" in config, "Missing evaluation key"
        assert "output" in config, "Missing output key"

        # Check experiment sub-keys
        assert "seed" in config["experiment"]
        assert "device" in config["experiment"]

        # Check model sub-keys
        assert "config_path" in config["model"]
        assert "weights_path" in config["model"]

    def test_model_config_yaml_valid(self, xai_root):
        """Load model_config_RawNet.yaml and verify required keys."""
        config_path = xai_root / "config" / "model_config_RawNet.yaml"
        assert config_path.exists(), "Config not found: {}".format(config_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model = config["model"]
        assert model["nb_samp"] == 64600
        assert model["first_conv"] == 1024
        assert model["in_channels"] == 1
        assert model["nb_classes"] == 2
        assert model["gru_node"] == 1024
        assert model["nb_gru_layer"] == 3


class TestEERIntegration:
    """Integration tests for end-to-end EER verification (FOUND-01).

    These tests require GPU, pretrained weights, and ASVspoof5 dev data.
    They are slow and should be run with --runslow flag or directly.
    """

    @pytest.mark.slow
    def test_eer_matches_baseline(self, model_config_path, weights_path, device, xai_root):
        """FOUND-01: EER within expected range on ASVspoof5 dev.

        Published baseline for RawNet2 (DF pretrained) on ASVspoof5 dev: ~30-36% EER.
        We use a 5000-sample stratified subset for test speed.
        """
        from models.loader import load_rawnet2
        from evaluation.inference import run_inference
        from evaluation.eer import compute_eer

        # Load config
        config_path = xai_root / "config" / "experiment.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Use subset for test speed
        config["evaluation"]["max_samples"] = 5000

        # Load model
        model = load_rawnet2(model_config_path, weights_path, device=device, verbose=False)

        # Run inference
        labels, scores = run_inference(model, config, device)

        # Compute EER
        eer, threshold = compute_eer(labels, scores)

        # Expected range for ASVspoof5 dev with DF pretrained weights
        # Published ~36% EER on eval; dev expected ~30-36%
        assert 0.20 < eer < 0.50, (
            "EER {:.2f}% outside expected range [20%, 50%] for ASVspoof5 dev. "
            "Check score polarity or input preprocessing.".format(eer * 100)
        )
        print("EER Integration Test: {:.2f}% (expected 30-36%)".format(eer * 100))
