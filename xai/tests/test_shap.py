"""Tests for Phase 5 SHAP baseline track (xai/shap/ package + scripts/{run,validate}_shap).

Test classes map 1:1 to the new xai/shap/ modules:
    TestFeatureExtraction  -> xai/shap/features.py
    TestCache              -> xai/shap/cache.py
    TestClassifier         -> xai/shap/classifier.py
    TestKernelSHAP         -> xai/shap/explainer.py
    TestComparison         -> xai/shap/compare.py
    TestViz                -> xai/shap/viz.py
    TestValidation         -> xai/scripts/validate_shap.py

Wave-0 expectation: the xai.shap.* and xai.scripts.validate_shap imports will
raise ImportError until the corresponding plans land. This is the correct
initial state per the Phase 5 Wave-0 protocol (same pattern as Phase 4 Plan 01).
"""
import json
import pytest
import numpy as np
from pathlib import Path


class TestFeatureExtraction:
    """Tests for xai/shap/features.py (SHAP-01)."""

    def test_returns_60_plus_features(self, synthetic_audio):
        from xai.shap.features import extract_acoustic_features
        audio, sr = synthetic_audio
        vec, names = extract_acoustic_features(audio, sr=sr)
        assert len(names) >= 60, f'SHAP-01 requires >=60 features, got {len(names)}'
        assert vec.shape == (len(names),)
        assert np.all(np.isfinite(vec)), 'all features must be finite (NaN-safe)'

    def test_feature_names_deterministic_order(self, synthetic_audio):
        """Feature name order must be stable across calls so the cache hash is reproducible."""
        from xai.shap.features import extract_acoustic_features
        audio, sr = synthetic_audio
        _, names1 = extract_acoustic_features(audio, sr=sr)
        _, names2 = extract_acoustic_features(audio, sr=sr)
        assert names1 == names2

    def test_nan_safe_on_short_clip(self):
        """parselmouth returns NaN on very short/unvoiced clips; extractor must coerce to 0.0."""
        from xai.shap.features import extract_acoustic_features
        short_audio = np.zeros(800, dtype=np.float32)  # 50 ms of silence
        vec, names = extract_acoustic_features(short_audio, sr=16000)
        assert np.all(np.isfinite(vec)), 'silence produced non-finite features'

    def test_required_feature_names_present(self, synthetic_audio):
        """D-01 concept->features mapping depends on these exact names."""
        from xai.shap.features import extract_acoustic_features
        audio, sr = synthetic_audio
        _, names = extract_acoustic_features(audio, sr=sr)
        required = {'HNR_mean', 'HNR_var', 'F0_mean', 'F0_std',
                    'spectral_flux_var', 'spectral_centroid_var',
                    'energy_envelope_var', 'pause_rate'}
        missing = required - set(names)
        assert not missing, f'D-01 mapping references features missing from extractor: {missing}'


class TestCache:
    """Tests for xai/shap/cache.py (feature caching by config-hash; SHAP-01 amortisation)."""

    def test_cache_hit(self, tmp_path, fake_shap_config):
        """Second call with same config loads from cache (no extractor re-invocation)."""
        from xai.shap.cache import load_or_extract_features
        cfg = fake_shap_config['cfg']
        calls = {'n': 0}

        def extractor(clip_id):
            calls['n'] += 1
            rng = np.random.RandomState(hash(clip_id) & 0xffffffff)
            return rng.randn(64).astype(np.float64), [f'feat_{i}' for i in range(64)], int(hash(clip_id) & 1)

        clip_ids = [f'clip_{i:04d}' for i in range(10)]
        X1, y1, n1 = load_or_extract_features(cfg, clip_ids, extractor=extractor, output_dir=tmp_path)
        first_call_count = calls['n']
        X2, y2, n2 = load_or_extract_features(cfg, clip_ids, extractor=extractor, output_dir=tmp_path)
        assert calls['n'] == first_call_count, 'cache HIT should not call extractor again'
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
        assert n1 == n2

    def test_classifier_section_edit_preserves_feature_cache(self, tmp_path, fake_shap_config):
        """Changing classifier config must NOT invalidate the feature cache."""
        from xai.shap.cache import load_or_extract_features
        cfg = fake_shap_config['cfg']
        calls = {'n': 0}

        def extractor(clip_id):
            calls['n'] += 1
            return np.zeros(64), [f'feat_{i}' for i in range(64)], 0

        clip_ids = [f'clip_{i:04d}' for i in range(5)]
        load_or_extract_features(cfg, clip_ids, extractor=extractor, output_dir=tmp_path)
        first = calls['n']
        cfg2 = dict(cfg)
        cfg2['classifier'] = dict(cfg['classifier'])
        cfg2['classifier']['rf'] = dict(cfg['classifier']['rf'])
        cfg2['classifier']['rf']['n_estimators'] = 500
        load_or_extract_features(cfg2, clip_ids, extractor=extractor, output_dir=tmp_path)
        assert calls['n'] == first, 'classifier section changed -> feature cache must survive'

    def test_features_section_edit_invalidates(self, tmp_path, fake_shap_config):
        """Changing features config MUST invalidate the cache (force re-extraction)."""
        from xai.shap.cache import load_or_extract_features
        cfg = fake_shap_config['cfg']
        calls = {'n': 0}

        def extractor(clip_id):
            calls['n'] += 1
            return np.zeros(64), [f'feat_{i}' for i in range(64)], 0

        clip_ids = [f'clip_{i:04d}' for i in range(5)]
        load_or_extract_features(cfg, clip_ids, extractor=extractor, output_dir=tmp_path)
        first = calls['n']
        cfg2 = dict(cfg)
        cfg2['features'] = dict(cfg['features'])
        cfg2['features']['n_mfcc'] = 20  # different feature set -> re-extract
        load_or_extract_features(cfg2, clip_ids, extractor=extractor, output_dir=tmp_path)
        assert calls['n'] > first, 'features section changed -> cache must be invalidated'


class TestClassifier:
    """Tests for xai/shap/classifier.py (SHAP-02)."""

    def test_rf_better_than_chance(self, synthetic_acoustic_features):
        from xai.shap.classifier import train_and_evaluate
        data = synthetic_acoustic_features
        out = train_and_evaluate(data['X'], data['y'], seed=42)
        assert out['results']['rf']['auc'] > 0.5, 'RF must beat chance on separable synthetic data'

    def test_svm_proba(self, synthetic_acoustic_features):
        """SHAP-02 explicitly requires SVC(probability=True) so KernelSHAP works downstream."""
        from xai.shap.classifier import train_and_evaluate
        data = synthetic_acoustic_features
        out = train_and_evaluate(data['X'], data['y'], seed=42)
        svm = out['models']['svm']
        proba = svm.predict_proba(out['X_test'])
        assert proba.shape[1] == 2
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_class_weight_balanced(self, synthetic_acoustic_features):
        """class_weight='balanced' must actually propagate into the fitted estimators."""
        from xai.shap.classifier import train_and_evaluate
        data = synthetic_acoustic_features
        out = train_and_evaluate(data['X'], data['y'], seed=42)
        assert out['models']['rf'].class_weight == 'balanced'
        assert out['models']['svm'].class_weight == 'balanced'

    def test_best_classifier_selected(self, synthetic_acoustic_features):
        """best_classifier must be one of 'rf' / 'svm' and match the higher AUC (ties -> rf)."""
        from xai.shap.classifier import train_and_evaluate
        data = synthetic_acoustic_features
        out = train_and_evaluate(data['X'], data['y'], seed=42)
        best = out['results']['best_classifier']
        assert best in ('rf', 'svm')
        rf_auc, svm_auc = out['results']['rf']['auc'], out['results']['svm']['auc']
        expected = 'rf' if rf_auc >= svm_auc else 'svm'
        assert best == expected


class TestKernelSHAP:
    """Tests for xai/shap/explainer.py (SHAP-03)."""

    def test_binary_output_format(self, synthetic_acoustic_features):
        """shap.KernelExplainer returns list [class0, class1] for binary classifier; must be handled."""
        from xai.shap.classifier import train_and_evaluate
        from xai.shap.explainer import compute_kernel_shap
        data = synthetic_acoustic_features
        out = train_and_evaluate(data['X'], data['y'], seed=42)
        result = compute_kernel_shap(
            out['models'][out['results']['best_classifier']],
            out['X_train'], out['X_test'][:20],
            feature_names=data['feature_names'],
            target_class=1, n_background=10, n_eval_cap=20, nsamples=64, seed=42,
        )
        assert result['shap_values'].shape == (20, 64)
        assert result['mean_abs_shap'].shape == (64,)
        assert len(result['ranked_features']) == 64

    def test_ranking_deterministic(self, synthetic_acoustic_features):
        """Top-15 ranking must be reproducible with the same seed."""
        from xai.shap.classifier import train_and_evaluate
        from xai.shap.explainer import compute_kernel_shap
        data = synthetic_acoustic_features
        out = train_and_evaluate(data['X'], data['y'], seed=42)
        r1 = compute_kernel_shap(out['models']['rf'], out['X_train'], out['X_test'][:20],
                                 feature_names=data['feature_names'], n_background=10,
                                 n_eval_cap=20, nsamples=64, seed=42)
        r2 = compute_kernel_shap(out['models']['rf'], out['X_train'], out['X_test'][:20],
                                 feature_names=data['feature_names'], n_background=10,
                                 n_eval_cap=20, nsamples=64, seed=42)
        top15_1 = [n for n, _ in r1['ranked_features'][:15]]
        top15_2 = [n for n, _ in r2['ranked_features'][:15]]
        assert top15_1 == top15_2

    def test_eval_cap_respected(self, synthetic_acoustic_features):
        """n_eval_cap must bound the runtime -- larger eval arrays get subsampled."""
        from xai.shap.classifier import train_and_evaluate
        from xai.shap.explainer import compute_kernel_shap
        data = synthetic_acoustic_features
        out = train_and_evaluate(data['X'], data['y'], seed=42)
        result = compute_kernel_shap(out['models']['rf'], out['X_train'], out['X_test'],
                                     feature_names=data['feature_names'], n_background=10,
                                     n_eval_cap=15, nsamples=64, seed=42)
        assert result['n_eval_used'] == 15


class TestComparison:
    """Tests for xai/shap/compare.py (SHAP-04)."""

    def test_spearman_sanity(self):
        """Perfect monotonic agreement -> rho = 1."""
        from xai.shap.compare import compare_shap_tcav
        mean_abs_shap = {'HNR_mean': 0.8, 'HNR_var': 0.6,
                         'F0_std': 0.5, 'F0_mean': 0.3}
        tcav = {
            'systems_analyzed': ['A09'],
            'per_system_scores': {'A09': {'breathiness': {'post_gru': {'mean_score': 0.8, 'significant': True}},
                                          'pitch_monotony': {'post_gru': {'mean_score': 0.5, 'significant': True}}}},
            'concept_signatures': {}, 'transferability': {},
            'n_significant': 2, 'fdr_method': 'benjamini_hochberg',
            'alpha': 0.05, 'n_tests_total': 2,
        }
        mapping = {'breathiness': ['HNR_mean', 'HNR_var'],
                   'pitch_monotony': ['F0_std', 'F0_mean']}
        result = compare_shap_tcav(mean_abs_shap, tcav, mapping, rho_confirm=0.7, rho_challenge=0.3)
        assert result['per_concept_rho']['breathiness']['rho'] == pytest.approx(1.0)
        assert result['per_concept_rho']['pitch_monotony']['rho'] == pytest.approx(1.0)
        assert result['comparison_status'] in ('confirms', 'inconclusive')

    def test_status_thresholds(self):
        """D-02/D-03 thresholds classify status correctly."""
        from xai.shap.compare import compare_shap_tcav
        mean_abs_shap_confirm = {'a': 0.9, 'b': 0.1, 'c': 0.8, 'd': 0.2}
        mapping = {'C1': ['a', 'b'], 'C2': ['c', 'd']}
        tcav_confirm = {'systems_analyzed': ['S'],
                        'per_system_scores': {'S': {'C1': {'L': {'mean_score': 0.9, 'significant': True}},
                                                    'C2': {'L': {'mean_score': 0.7, 'significant': True}}}},
                        'concept_signatures': {}, 'transferability': {},
                        'n_significant': 2, 'fdr_method': 'benjamini_hochberg',
                        'alpha': 0.05, 'n_tests_total': 2}
        r = compare_shap_tcav(mean_abs_shap_confirm, tcav_confirm, mapping,
                              rho_confirm=0.7, rho_challenge=0.3)
        assert r['comparison_status'] in ('confirms', 'inconclusive')
        assert r['per_concept_rho']['C1']['rho'] == pytest.approx(1.0)

    def test_nan_rho_handled(self):
        """Too-few mapped features -> rho = NaN, status = 'inconclusive', no crash."""
        from xai.shap.compare import compare_shap_tcav
        mean_abs_shap = {'a': 0.5}  # only one feature
        mapping = {'C1': ['a', 'b']}  # 'b' not present -> only 1 mapped -> NaN
        tcav = {'systems_analyzed': ['S'],
                'per_system_scores': {'S': {'C1': {'L': {'mean_score': 0.5, 'significant': False}}}},
                'concept_signatures': {}, 'transferability': {},
                'n_significant': 0, 'fdr_method': 'benjamini_hochberg',
                'alpha': 0.05, 'n_tests_total': 1}
        r = compare_shap_tcav(mean_abs_shap, tcav, mapping, rho_confirm=0.7, rho_challenge=0.3)
        assert r['comparison_status'] == 'inconclusive'
        assert np.isnan(r['per_concept_rho']['C1']['rho'])

    def test_schema_drift_detected(self, tmp_path):
        """Malformed per_system_results.json must raise KeyError (Pitfall 7)."""
        from xai.shap.compare import load_tcav_results
        bad = tmp_path / 'bad.json'
        bad.write_text(json.dumps({'some_key': 'some_value'}))
        with pytest.raises(KeyError):
            load_tcav_results(str(bad))


class TestViz:
    """Tests for xai/shap/viz.py (FIG-06 under D-04)."""

    def test_fig06_produced(self, tmp_path):
        """plot_shap_importance writes a PNG > 50 KB (gate 4 in validate_shap)."""
        from xai.shap.viz import plot_shap_importance
        ranked = [(f'feat_{i}', 0.1 - 0.001 * i) for i in range(20)]
        per_concept_rho = {'breathiness': {'rho': 0.82, 'pval': 0.01, 'n_mapped': 2}}
        out = tmp_path / 'shap_importance.png'
        plot_shap_importance(ranked_features=ranked,
                             per_concept_rho=per_concept_rho,
                             overall_rho=0.71,
                             comparison_status='confirms',
                             output_path=str(out), top_k=15)
        assert out.exists()
        assert out.stat().st_size > 50_000, f'FIG-06 must exceed 50 KB, got {out.stat().st_size}'


class TestValidation:
    """Tests for xai/scripts/validate_shap.py (integrates all gates)."""

    def test_exit_1_on_missing_file(self, tmp_path):
        from xai.scripts.validate_shap import validate_shap
        assert validate_shap(str(tmp_path / 'nonexistent.json')) == 1

    def test_exit_0_on_valid_results(self, tmp_path, synthetic_shap_results, monkeypatch):
        from xai.scripts import validate_shap as mod
        results_path = tmp_path / 'shap_results.json'
        results_path.write_text(json.dumps(synthetic_shap_results))
        fig_dir = tmp_path / 'results' / 'shap' / 'figures'
        fig_dir.mkdir(parents=True)
        (fig_dir / 'shap_importance.png').write_bytes(b'\x89PNG' + b'\x00' * 60_000)
        monkeypatch.setattr(mod, 'XAI_ROOT', tmp_path)
        assert mod.validate_shap(str(results_path)) == 0

    def test_exit_1_when_auc_below_threshold(self, tmp_path, synthetic_shap_results, monkeypatch):
        from xai.scripts import validate_shap as mod
        bad = dict(synthetic_shap_results)
        bad['classifier'] = dict(synthetic_shap_results['classifier'])
        bad['classifier']['rf'] = {**synthetic_shap_results['classifier']['rf'], 'auc': 0.50}
        bad['classifier']['svm'] = {**synthetic_shap_results['classifier']['svm'], 'auc': 0.50}
        results_path = tmp_path / 'shap_results.json'
        results_path.write_text(json.dumps(bad))
        monkeypatch.setattr(mod, 'XAI_ROOT', tmp_path)
        assert mod.validate_shap(str(results_path)) == 1
