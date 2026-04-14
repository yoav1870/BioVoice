"""Unit tests for Phase 2: Acoustic concept measures, config loading, and manifest I/O.

Covers requirements CONC-01 (acoustic measures), CONC-02 (concept YAML config),
CONC-03 (manifest I/O), CONC-04 (measure correctness), CONC-05 (no hardcoded thresholds).
"""

import os
import tempfile

import numpy as np
import pytest
import yaml

from xai.concepts.measures import (
    compute_mean_hnr,
    compute_f0_std,
    compute_spectral_flux_var,
    compute_energy_envelope_var,
)
from xai.concepts.manifest import write_manifest, read_manifest


class TestAcousticMeasures:
    """Tests for the four acoustic measure functions (CONC-01, CONC-04)."""

    def test_compute_mean_hnr_returns_float(self, synthetic_audio):
        """HNR of a pure 440Hz sine wave should return a positive float."""
        audio, sr = synthetic_audio
        result = compute_mean_hnr(audio, sr)
        assert isinstance(result, float), "Expected float, got {}".format(type(result))
        assert result > 0, "Expected HNR > 0 for pure sine, got {}".format(result)

    def test_compute_mean_hnr_noise_lower_than_sine(self, synthetic_audio):
        """HNR of white noise should be lower than HNR of a pure sine wave."""
        audio, sr = synthetic_audio
        rng = np.random.RandomState(42)
        noise = rng.randn(len(audio))
        hnr_sine = compute_mean_hnr(audio, sr)
        hnr_noise = compute_mean_hnr(noise, sr)
        assert hnr_sine > hnr_noise, (
            "HNR of sine ({:.2f} dB) should exceed HNR of noise ({:.2f} dB)".format(
                hnr_sine, hnr_noise
            )
        )

    def test_compute_f0_std_sine_near_zero(self, synthetic_audio):
        """F0 std of a pure 440Hz sine wave should be near zero (constant pitch)."""
        audio, sr = synthetic_audio
        result = compute_f0_std(audio, sr)
        assert isinstance(result, float), "Expected float, got {}".format(type(result))
        # Pure sine has constant F0 -- std should be very small (< 5 Hz)
        assert result < 5.0, (
            "F0 std of pure sine should be near zero, got {:.2f} Hz".format(result)
        )

    def test_compute_f0_std_returns_float(self):
        """F0 std on a chirp-like signal (frequency-modulated) returns float >= 0."""
        sr = 16000
        t = np.arange(2 * sr) / sr  # 2-second signal for reliable pitch detection
        # Linear chirp: 200 Hz to 600 Hz over 2 seconds
        f_inst = 200 + 200 * t
        audio = np.sin(2 * np.pi * np.cumsum(f_inst) / sr).astype(np.float64)
        result = compute_f0_std(audio, sr)
        assert isinstance(result, float), "Expected float, got {}".format(type(result))
        assert result >= 0.0, "Expected F0 std >= 0, got {}".format(result)

    def test_compute_spectral_flux_var_returns_float(self, synthetic_audio):
        """Spectral flux variance of a 1-second sine wave returns a float >= 0."""
        audio, sr = synthetic_audio
        result = compute_spectral_flux_var(audio, sr)
        assert isinstance(result, float), "Expected float, got {}".format(type(result))
        assert result >= 0.0, "Expected spectral flux var >= 0, got {}".format(result)

    def test_compute_spectral_flux_var_chirp_higher_than_silent(self):
        """Spectral flux variance of a frequency-varying chirp exceeds near-silence.

        A chirp rapidly sweeps across frequencies (high flux changes frame-to-frame),
        while a near-silent signal has minimal spectral flux.
        """
        sr = 16000
        t = np.arange(sr) / sr
        # Chirp from 100 Hz to 4000 Hz (large spectral sweep = high flux variance)
        f_inst = 100 + 3900 * t
        chirp = np.sin(2 * np.pi * np.cumsum(f_inst) / sr).astype(np.float64)
        # Near-silent signal: very small amplitude random noise
        silence = np.zeros(sr, dtype=np.float64)
        sfv_chirp = compute_spectral_flux_var(chirp, sr)
        sfv_silence = compute_spectral_flux_var(silence, sr)
        assert sfv_chirp > sfv_silence, (
            "Chirp sfv ({:.6f}) should exceed silence sfv ({:.6f})".format(
                sfv_chirp, sfv_silence
            )
        )

    def test_compute_energy_envelope_var_returns_float(self, synthetic_audio):
        """Energy envelope variance of a sine wave returns float >= 0."""
        audio, sr = synthetic_audio
        result = compute_energy_envelope_var(audio, sr)
        assert isinstance(result, float), "Expected float, got {}".format(type(result))
        assert result >= 0.0, "Expected energy envelope var >= 0, got {}".format(result)

    def test_compute_energy_envelope_var_steady_lower_than_modulated(self):
        """Energy envelope variance of steady sine is lower than amplitude-modulated signal.

        A sine with constant amplitude should have lower energy envelope variance
        than one with strong amplitude modulation (tremolo effect).
        Note: RMS frames from a 440Hz sine have non-zero variance due to phase
        alignment at frame boundaries; threshold is relative, not absolute.
        """
        sr = 16000
        t = np.arange(sr) / sr
        # Steady sine: constant amplitude
        steady = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        # Amplitude-modulated sine: amplitude varies from 0.1 to 1.0 at 5 Hz
        mod = (0.55 + 0.45 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 440 * t)
        mod = mod.astype(np.float64)
        ev_steady = compute_energy_envelope_var(steady, sr)
        ev_modulated = compute_energy_envelope_var(mod, sr)
        assert ev_steady < ev_modulated, (
            "Steady sine env var ({:.2e}) should be below modulated ({:.2e})".format(
                ev_steady, ev_modulated
            )
        )


class TestManifestIO:
    """Tests for manifest write/read I/O (CONC-03)."""

    def test_manifest_write_read_roundtrip(self, tmp_path):
        """write_manifest then read_manifest returns same data as strings."""
        rows = [
            {"audio_id": "clip_001", "speaker_id": "spk_A", "measure_value": "12.34"},
            {"audio_id": "clip_002", "speaker_id": "spk_B", "measure_value": "5.67"},
        ]
        columns = ["audio_id", "speaker_id", "measure_value"]
        output_path = str(tmp_path / "test_manifest.csv")

        write_manifest(rows, output_path, columns)
        result = read_manifest(output_path)

        assert len(result) == 2, "Expected 2 rows, got {}".format(len(result))
        assert result[0]["audio_id"] == "clip_001"
        assert result[0]["speaker_id"] == "spk_A"
        assert result[0]["measure_value"] == "12.34"
        assert result[1]["audio_id"] == "clip_002"

    def test_manifest_creates_parent_dirs(self, tmp_path):
        """write_manifest creates intermediate directories if they don't exist."""
        nested_path = str(tmp_path / "nested" / "deep" / "manifest.csv")
        rows = [{"audio_id": "a", "speaker_id": "b", "measure_value": "1.0"}]
        columns = ["audio_id", "speaker_id", "measure_value"]

        # Should not raise -- must create parent dirs automatically
        write_manifest(rows, nested_path, columns)
        assert os.path.isfile(nested_path), "Manifest file should exist after write"

    def test_manifest_preserves_column_order(self, tmp_path):
        """CSV header respects the columns list order."""
        rows = [{"audio_id": "x", "measure_value": "0.5", "speaker_id": "s1"}]
        columns = ["measure_value", "audio_id", "speaker_id"]
        output_path = str(tmp_path / "ordered.csv")
        write_manifest(rows, output_path, columns)

        with open(output_path, "r") as f:
            header = f.readline().strip()
        assert header == "measure_value,audio_id,speaker_id", (
            "Expected specific column order, got: {}".format(header)
        )


class TestConceptConfig:
    """Tests for concepts.yaml structure and values (CONC-02, CONC-05)."""

    def test_config_loads_four_concepts(self, concept_config_path):
        """concepts.yaml loads with yaml.safe_load and has exactly 4 concept entries."""
        with open(concept_config_path) as f:
            cfg = yaml.safe_load(f)
        assert "concepts" in cfg, "Config must have 'concepts' key"
        assert len(cfg["concepts"]) == 4, (
            "Expected 4 concepts, got {}: {}".format(
                len(cfg["concepts"]), list(cfg["concepts"].keys())
            )
        )

    def test_config_has_required_fields(self, concept_config_path):
        """Each concept must have measure, direction, threshold, min_clips, min_speakers."""
        required_fields = {"measure", "direction", "threshold", "min_clips", "min_speakers"}
        with open(concept_config_path) as f:
            cfg = yaml.safe_load(f)
        for concept_name, concept_cfg in cfg["concepts"].items():
            missing = required_fields - set(concept_cfg.keys())
            assert not missing, (
                "Concept '{}' missing required fields: {}".format(concept_name, missing)
            )

    def test_config_negative_control(self, concept_config_path):
        """Config has 'negative_control' section with 'n_clips' and 'seed' keys."""
        with open(concept_config_path) as f:
            cfg = yaml.safe_load(f)
        assert "negative_control" in cfg, "Config must have 'negative_control' section"
        nc = cfg["negative_control"]
        assert "n_clips" in nc, "'negative_control' must have 'n_clips'"
        assert "seed" in nc, "'negative_control' must have 'seed'"
        assert nc["n_clips"] == 200, "Expected n_clips=200, got {}".format(nc["n_clips"])
        assert nc["seed"] == 42, "Expected seed=42, got {}".format(nc["seed"])

    def test_config_concept_names(self, concept_config_path):
        """Config contains the four expected concept names."""
        expected_names = {"breathiness", "pitch_monotony", "spectral_smoothness", "temporal_regularity"}
        with open(concept_config_path) as f:
            cfg = yaml.safe_load(f)
        actual_names = set(cfg["concepts"].keys())
        assert actual_names == expected_names, (
            "Expected concepts {}, got {}".format(expected_names, actual_names)
        )

    def test_config_threshold_values(self, concept_config_path):
        """Config contains the expected pilot threshold values."""
        with open(concept_config_path) as f:
            cfg = yaml.safe_load(f)
        concepts = cfg["concepts"]
        assert concepts["breathiness"]["threshold"] == pytest.approx(7.8)
        assert concepts["pitch_monotony"]["threshold"] == pytest.approx(25.8)
        assert concepts["spectral_smoothness"]["threshold"] == pytest.approx(0.016)
        assert concepts["temporal_regularity"]["threshold"] == pytest.approx(0.0005)


# ---------------------------------------------------------------------------
# Phase 02 Plan 02 tests: filtering, negative control, disjointness
# ---------------------------------------------------------------------------

from xai.concepts.filtering import (
    load_features_csv,
    filter_by_threshold,
    build_all_concept_sets,
)
from xai.concepts.negative_control import (
    generate_negative_control,
    build_negative_control,
)


class TestFiltering:
    """Tests for threshold-based concept filtering (CONC-01, CONC-06)."""

    def test_filter_by_threshold_below(self, mock_features_csv):
        """filter_by_threshold with direction below keeps only rows below threshold."""
        rows = load_features_csv(mock_features_csv)
        # mean_hnr values: 5.0 + i*0.8 for i in 0..19
        # Threshold 7.8: rows 0-3 pass (5.0, 5.8, 6.6, 7.4)
        result = filter_by_threshold(rows, "mean_hnr", 7.8, "below")
        assert len(result) == 4, "Expected 4 rows below 7.8, got {}".format(len(result))
        for r in result:
            assert r["mean_hnr"] < 7.8, "Row has HNR {} >= 7.8".format(r["mean_hnr"])

    def test_filter_by_threshold_above(self, mock_features_csv):
        """filter_by_threshold with direction above keeps only rows above threshold."""
        rows = load_features_csv(mock_features_csv)
        result = filter_by_threshold(rows, "mean_hnr", 15.0, "above")
        assert len(result) > 0, "Expected some rows above 15.0"
        for r in result:
            assert r["mean_hnr"] > 15.0, "Row has HNR {} <= 15.0".format(r["mean_hnr"])

    def test_filter_by_threshold_sorted(self, mock_features_csv):
        """Filtered results are sorted by audio_id for determinism."""
        rows = load_features_csv(mock_features_csv)
        result = filter_by_threshold(rows, "mean_hnr", 20.0, "below")
        audio_ids = [r["audio_id"] for r in result]
        assert audio_ids == sorted(audio_ids), "Results not sorted by audio_id"

    def test_filter_by_threshold_invalid_direction(self, mock_features_csv):
        """filter_by_threshold raises ValueError for unknown direction."""
        rows = load_features_csv(mock_features_csv)
        with pytest.raises(ValueError, match="Unknown direction"):
            filter_by_threshold(rows, "mean_hnr", 10.0, "invalid")

    def test_build_all_concept_sets_creates_manifests(
        self, mock_features_csv, concept_config_path, tmp_path
    ):
        """build_all_concept_sets writes per-concept manifest.csv files."""
        output_dir = str(tmp_path / "concepts")
        results = build_all_concept_sets(
            features_csv_path=mock_features_csv,
            concepts_yaml_path=concept_config_path,
            output_dir=output_dir,
        )
        expected = {"breathiness", "pitch_monotony", "spectral_smoothness", "temporal_regularity"}
        assert set(results.keys()) == expected, "Unexpected concept names: {}".format(set(results.keys()))
        for concept_name in expected:
            manifest_path = os.path.join(output_dir, concept_name, "manifest.csv")
            assert os.path.isfile(manifest_path), (
                "Expected manifest.csv at {}".format(manifest_path)
            )


class TestNegativeControl:
    """Tests for negative control concept generation (CONC-07)."""

    def test_negative_control_deterministic(self, mock_features_csv):
        """generate_negative_control with same seed produces identical output."""
        rows = load_features_csv(mock_features_csv)
        result1 = generate_negative_control(rows, n_clips=10, seed=42)
        result2 = generate_negative_control(rows, n_clips=10, seed=42)
        ids1 = [r["audio_id"] for r in result1]
        ids2 = [r["audio_id"] for r in result2]
        assert ids1 == ids2, "Same seed should produce identical results"

    def test_negative_control_excludes_ids(self, mock_features_csv):
        """generate_negative_control excludes specified audio_ids from pool."""
        rows = load_features_csv(mock_features_csv)
        excluded = set(r["audio_id"] for r in rows[:5])
        result = generate_negative_control(rows, n_clips=10, seed=42, exclude_audio_ids=excluded)
        result_ids = set(r["audio_id"] for r in result)
        overlap = result_ids & excluded
        assert len(overlap) == 0, "Excluded IDs appear in result: {}".format(overlap)

    def test_negative_control_respects_n_clips(self, mock_features_csv):
        """generate_negative_control returns exactly n_clips when pool is large enough."""
        rows = load_features_csv(mock_features_csv)
        result = generate_negative_control(rows, n_clips=10, seed=42)
        assert len(result) == 10, "Expected 10 clips, got {}".format(len(result))


class TestDisjointness:
    """Tests for train/dev/eval disjointness invariants."""

    def test_disjointness_train_prefixes(self, mock_features_csv):
        """All speaker_ids in mock data start with T_ (train partition)."""
        rows = load_features_csv(mock_features_csv)
        bad_speakers = [r["speaker_id"] for r in rows if not r["speaker_id"].startswith("T_")]
        assert len(bad_speakers) == 0, "Found non-T_ speakers: {}".format(bad_speakers)

    def test_reproducibility_filtering(
        self, mock_features_csv, concept_config_path, tmp_path
    ):
        """Running build_all_concept_sets twice produces identical manifests."""
        import yaml
        output_dir_1 = str(tmp_path / "run1")
        output_dir_2 = str(tmp_path / "run2")
        build_all_concept_sets(mock_features_csv, concept_config_path, output_dir_1)
        build_all_concept_sets(mock_features_csv, concept_config_path, output_dir_2)
        with open(concept_config_path) as f:
            cfg = yaml.safe_load(f)
        for concept_name in cfg["concepts"]:
            m1 = os.path.join(output_dir_1, concept_name, "manifest.csv")
            m2 = os.path.join(output_dir_2, concept_name, "manifest.csv")
            if os.path.isfile(m1) and os.path.isfile(m2):
                # Manual content comparison (filecmp not in allowed_prefixes)
                with open(m1) as f1, open(m2) as f2:
                    content1 = f1.read()
                    content2 = f2.read()
                assert content1 == content2, (
                    "Manifest for {} differs between runs".format(concept_name)
                )
            elif os.path.isfile(m1) or os.path.isfile(m2):
                assert False, "Concept {} manifest missing in one run".format(concept_name)

