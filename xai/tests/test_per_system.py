"""Tests for per-synthesis-system TCAV analysis (Phase 4)."""
import pytest
import numpy as np
import json


class TestProtocolParsing:
    """Tests for parse_dev_protocol_by_system()."""

    def test_groups_spoof_clips_by_system(self, fake_protocol_file):
        from xai.scripts.run_per_system_tcav import parse_dev_protocol_by_system
        systems, bonafide = parse_dev_protocol_by_system(fake_protocol_file)
        assert "A09" in systems
        assert "A10" in systems
        assert "A12" in systems
        assert len(systems["A09"]) == 2
        assert len(systems["A10"]) == 3
        assert len(systems["A12"]) == 1

    def test_excludes_bonafide_clips(self, fake_protocol_file):
        from xai.scripts.run_per_system_tcav import parse_dev_protocol_by_system
        systems, bonafide = parse_dev_protocol_by_system(fake_protocol_file)
        assert len(bonafide) == 2
        all_spoof_ids = []
        for clips in systems.values():
            all_spoof_ids.extend(c["audio_id"] for c in clips)
        for b in bonafide:
            assert b["audio_id"] not in all_spoof_ids

    def test_handles_space_delimited_protocol(self, fake_protocol_file):
        from xai.scripts.run_per_system_tcav import parse_dev_protocol_by_system
        systems, bonafide = parse_dev_protocol_by_system(fake_protocol_file)
        assert len(systems) > 0

    def test_returns_audio_id_and_speaker_id(self, fake_protocol_file):
        from xai.scripts.run_per_system_tcav import parse_dev_protocol_by_system
        systems, _ = parse_dev_protocol_by_system(fake_protocol_file)
        clip = systems["A09"][0]
        assert "audio_id" in clip
        assert "speaker_id" in clip
        assert clip["audio_id"].startswith("D_")


class TestSystemExclusion:
    """Tests for check_system_sample_counts() -- SYS-04."""

    def test_excludes_systems_below_threshold(self):
        from xai.scripts.run_per_system_tcav import check_system_sample_counts
        systems = {
            "A09": [{"audio_id": "D_%05d" % i} for i in range(100)],
            "A10": [{"audio_id": "D_%05d" % i} for i in range(30)],
            "A12": [{"audio_id": "D_%05d" % i} for i in range(200)],
        }
        valid, excluded = check_system_sample_counts(systems, min_samples=50)
        assert "A09" in valid
        assert "A12" in valid
        assert "A10" in excluded
        assert excluded["A10"]["count"] == 30

    def test_exclusion_logged(self, capsys):
        from xai.scripts.run_per_system_tcav import check_system_sample_counts
        systems = {
            "A10": [{"audio_id": "D_%05d" % i} for i in range(10)],
        }
        check_system_sample_counts(systems, min_samples=50)
        captured = capsys.readouterr()
        assert "EXCLUDED" in captured.out
        assert "A10" in captured.out
        assert "10" in captured.out

    def test_all_systems_valid_when_above_threshold(self):
        from xai.scripts.run_per_system_tcav import check_system_sample_counts
        systems = {
            "A09": [{"audio_id": "D_%05d" % i} for i in range(60)],
            "A10": [{"audio_id": "D_%05d" % i} for i in range(70)],
        }
        valid, excluded = check_system_sample_counts(systems, min_samples=50)
        assert len(valid) == 2
        assert len(excluded) == 0


class TestFDRCorrection:
    """Tests for fdr_bh_correction() in stats.py."""

    def test_no_significant_when_all_pvals_high(self):
        from xai.tcav.stats import fdr_bh_correction
        pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        result = fdr_bh_correction(pvals, alpha=0.05)
        assert result["n_significant"] == 0
        assert not np.any(result["rejected"])

    def test_known_bh_example(self):
        """BH example: sorted pvals 0.001,0.008,0.039,0.041,0.23 alpha=0.05 n=5.
        BH critical: 0.01,0.02,0.03,0.04,0.05. Reject k=2 (indices 1,3)."""
        from xai.tcav.stats import fdr_bh_correction
        pvals = np.array([0.041, 0.001, 0.23, 0.008, 0.039])
        result = fdr_bh_correction(pvals, alpha=0.05)
        assert result["n_significant"] == 2
        assert result["rejected"][1] == True
        assert result["rejected"][3] == True
        assert result["rejected"][0] == False
        assert result["rejected"][2] == False
        assert result["rejected"][4] == False

    def test_nan_pvals_treated_as_nonsignificant(self):
        from xai.tcav.stats import fdr_bh_correction
        pvals = np.array([0.001, np.nan, 0.5])
        result = fdr_bh_correction(pvals, alpha=0.05)
        assert result["rejected"][1] == False

    def test_corrected_pvals_bounded_by_1(self):
        from xai.tcav.stats import fdr_bh_correction
        pvals = np.array([0.8, 0.9, 0.95])
        result = fdr_bh_correction(pvals, alpha=0.05)
        assert np.all(result["pvals_corrected"] <= 1.0)

    def test_all_significant_when_pvals_very_small(self):
        from xai.tcav.stats import fdr_bh_correction
        pvals = np.array([0.0001, 0.0002, 0.0003])
        result = fdr_bh_correction(pvals, alpha=0.05)
        assert result["n_significant"] == 3
        assert np.all(result["rejected"])

    def test_empty_pvals_returns_zero_significant(self):
        from xai.tcav.stats import fdr_bh_correction
        pvals = np.array([])
        result = fdr_bh_correction(pvals, alpha=0.05)
        assert result["n_significant"] == 0


class TestConceptSignatures:
    """Tests for per-system concept elevation detection -- SYS-02."""

    def test_elevated_concept_detected(self):
        """A concept significant at any layer for a system is in its significant_concepts."""
        from xai.scripts.run_per_system_tcav import identify_concept_signatures
        significance = {
            ("A09", "breathiness", "post_gru"): True,
            ("A09", "breathiness", "sinc_conv"): False,
            ("A09", "pitch_monotony", "post_gru"): False,
        }
        signatures = identify_concept_signatures(
            significance, systems=["A09"],
            concepts=["breathiness", "pitch_monotony"],
            layers=["sinc_conv", "post_gru"]
        )
        assert "breathiness" in signatures["A09"]["significant_concepts"]
        assert "pitch_monotony" not in signatures["A09"]["significant_concepts"]


class TestTransferability:
    """Tests for classify_concept_transferability() -- SYS-03."""

    def test_universal_concept_classification(self):
        from xai.scripts.run_per_system_tcav import classify_concept_transferability
        systems = ["A%02d" % i for i in range(9, 17)]
        significance = {}
        for s in systems:
            significance[(s, "breathiness", "post_gru")] = True
        result = classify_concept_transferability(
            significance, systems=systems,
            concepts=["breathiness"], layers=["post_gru"]
        )
        assert result["breathiness"]["classification"] == "universal"
        assert result["breathiness"]["n_significant_systems"] == 8

    def test_fingerprint_concept_classification(self):
        from xai.scripts.run_per_system_tcav import classify_concept_transferability
        systems = ["A%02d" % i for i in range(9, 17)]
        significance = {}
        for s in systems:
            significance[(s, "pitch_monotony", "post_gru")] = (s == "A09")
        result = classify_concept_transferability(
            significance, systems=systems,
            concepts=["pitch_monotony"], layers=["post_gru"]
        )
        assert result["pitch_monotony"]["classification"] == "system_specific"
        assert result["pitch_monotony"]["n_significant_systems"] == 1

    def test_intermediate_concept_classification(self):
        from xai.scripts.run_per_system_tcav import classify_concept_transferability
        systems = ["A%02d" % i for i in range(9, 17)]
        significance = {}
        sig_systems = systems[:4]
        for s in systems:
            significance[(s, "spectral_smoothness", "post_gru")] = (s in sig_systems)
        result = classify_concept_transferability(
            significance, systems=systems,
            concepts=["spectral_smoothness"], layers=["post_gru"]
        )
        assert result["spectral_smoothness"]["classification"] == "intermediate"
        assert result["spectral_smoothness"]["n_significant_systems"] == 4
