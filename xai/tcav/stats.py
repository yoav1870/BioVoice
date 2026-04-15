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
