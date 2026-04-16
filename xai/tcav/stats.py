"""Statistical significance testing for TCAV scores.

Implements Welch's t-test against random baseline TCAV scores with Bonferroni
correction and bootstrap 95% confidence intervals.

Reference: Kim et al. 2018 TCAV -- significance testing ensures findings are
publishable (random concepts should not be significant; real concepts should be).
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import List, Dict, Tuple


def test_significance(real_scores: List[float],
                      random_scores_per_baseline: List[List[float]],
                      n_pairs: int) -> Dict:
    """
    Test whether real concept TCAV scores significantly differ from random baselines.

    real_scores:               TCAV scores from the real concept (one per random
                               train/test split or repeated measurement)
    random_scores_per_baseline: list of score-lists, one per random baseline concept
                               (>= 10 baselines required for adequate DoF)
    n_pairs:                   total (concept, layer) pairs for Bonferroni correction
                               (5 concepts x 5 layers = 25)

    Returns: {
        'pval': float (raw two-sided p-value),
        'pval_corrected': float (Bonferroni-corrected p-value),
        'significant': bool (pval_corrected < 0.05),
        'ci_95': tuple (bootstrap 95% CI on mean of real_scores),
        'mean_score': float (mean of real_scores),
        't_stat': float (t statistic),
    }
    """
    # Flatten all random baseline scores into one distribution
    random_flat = [s for baseline in random_scores_per_baseline for s in baseline]

    # Welch's t-test (unequal variance): real concept vs. random baselines
    t_stat, pval = scipy_stats.ttest_ind(
        real_scores, random_flat, equal_var=False
    )

    # Bonferroni correction: multiply p-value by number of tests
    # Handle NaN from scipy when all values are identical (zero variance) -- treat as p=1.0
    pval_float = float(pval) if not np.isnan(pval) else 1.0
    pval_corrected = min(pval_float * n_pairs, 1.0)

    # Bootstrap 95% CI on real TCAV scores
    real_arr = np.array(real_scores)
    if len(real_arr) >= 2:
        bootstrap_result = scipy_stats.bootstrap(
            (real_arr,),
            statistic=np.mean,
            n_resamples=1000,
            confidence_level=0.95,
            method='percentile',
        )
        ci_95 = (float(bootstrap_result.confidence_interval.low),
                 float(bootstrap_result.confidence_interval.high))
    else:
        # Single score -- CI is just the score itself
        ci_95 = (float(real_arr[0]), float(real_arr[0]))

    return {
        'pval': pval_float,
        'pval_corrected': float(pval_corrected),
        'significant': bool(pval_corrected < 0.05),
        'ci_95': ci_95,
        'mean_score': float(np.mean(real_scores)),
        't_stat': float(t_stat),
    }


def fdr_bh_correction(pvals, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction for multiple hypothesis testing.

    Manual implementation -- scipy.stats.false_discovery_control requires
    scipy >= 1.11.0; project has scipy 1.10.1.

    Args:
        pvals: 1D numpy array of raw p-values
        alpha: desired FDR level (default 0.05)

    Returns:
        dict with keys:
        - rejected: np.ndarray of bool (True = significant after FDR)
        - pvals_corrected: np.ndarray of BH-adjusted p-values
        - n_significant: int count of rejected hypotheses
    """
    import numpy as np
    n = len(pvals)
    if n == 0:
        return {
            "rejected": np.array([], dtype=bool),
            "pvals_corrected": np.array([]),
            "n_significant": 0,
        }

    # NaN-safe: treat NaN as 1.0 (same convention as test_significance)
    pvals_clean = np.where(np.isnan(pvals), 1.0, pvals)

    # Sort p-values ascending
    sorted_idx = np.argsort(pvals_clean)
    sorted_pvals = pvals_clean[sorted_idx]

    # BH critical values: (rank / n) * alpha
    ranks = np.arange(1, n + 1)
    bh_critical = (ranks / n) * alpha

    # Compute step-up adjusted p-values (Yekutieli & Benjamini 1999 formula)
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_pvals[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_pvals[i] * n / (i + 1)
        )
    adjusted = np.minimum(adjusted, 1.0)

    # Find largest k where p(k) <= bh_critical(k)
    below_threshold = sorted_pvals <= bh_critical
    if not np.any(below_threshold):
        return {
            "rejected": np.zeros(n, dtype=bool),
            "pvals_corrected": adjusted,
            "n_significant": 0,
        }

    # All hypotheses with rank <= max_k are rejected
    max_k = int(np.max(np.where(below_threshold)[0]))
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_idx[:max_k + 1]] = True

    return {
        "rejected": rejected,
        "pvals_corrected": adjusted,
        "n_significant": int(np.sum(rejected)),
    }
