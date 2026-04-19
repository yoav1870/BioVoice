"""KernelSHAP explainer for the SHAP baseline track (Phase 5).

Computes mean-absolute SHAP per feature on a capped evaluation subset, using a
kmeans-reduced background (~100x faster than full X_train per shap docs).
Handles the binary-classifier output list [class0, class1] (Pitfall 4) and
the evaluation-cap runtime bound (Pitfall 1) from 05-RESEARCH.md.
"""
from typing import Dict, List, Union

import numpy as np
import shap


def compute_kernel_shap(clf,
                        X_train: np.ndarray,
                        X_eval: np.ndarray,
                        feature_names: List[str],
                        target_class: int = 1,
                        n_background: int = 50,
                        n_eval_cap: int = 500,
                        nsamples: Union[int, str] = 'auto',
                        seed: int = 42) -> Dict:
    """Compute KernelSHAP feature importances on a bounded evaluation subset.

    Parameters:
        clf: Fitted sklearn estimator with `predict_proba(X)` method (RF or SVC
             with probability=True). KernelExplainer is model-agnostic.
        X_train: (n_train, n_features) -- reduced to `n_background` representative
                 points via `shap.kmeans`.
        X_eval: (n_eval, n_features) -- evaluation set; subsampled to `n_eval_cap`
                if larger (Pitfall 1 runtime bound).
        feature_names: list[str] length n_features (order matches columns of X).
        target_class: integer class index for binary output list selection
                      (Pitfall 4). Default 1 = spoof class.
        n_background: K for shap.kmeans(X_train, K). 50 is the SHAP-docs default.
        n_eval_cap: maximum number of evaluation instances (bounds runtime).
        nsamples: passed through to shap.KernelExplainer.shap_values. 'auto' ->
                  2*d + 2048 per shap docs.
        seed: np.random.RandomState seed for reproducible eval subsampling.

    Returns a dict with:
        'shap_values':     np.ndarray (n_eval_used, n_features) -- target-class SHAP values
        'mean_abs_shap':   np.ndarray (n_features,)
        'ranked_features': list[(feature_name, mean_abs_shap)] descending
        'n_eval_used':     int
        'n_background':    int
    """
    # Pitfall 1: bound the evaluation set.
    rng = np.random.RandomState(seed)
    if len(X_eval) > n_eval_cap:
        idx = rng.choice(len(X_eval), size=n_eval_cap, replace=False)
        idx.sort()
        X_eval_used = X_eval[idx]
    else:
        X_eval_used = X_eval

    # Pitfall 1 mitigation: kmeans background (~100x faster than full X_train).
    background = shap.kmeans(X_train, n_background)

    # KernelExplainer is model-agnostic; works on both RF and SVC(probability=True).
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    shap_values = explainer.shap_values(X_eval_used, nsamples=nsamples)

    # Pitfall 4: binary classifier returns list [class0, class1]; select target.
    if isinstance(shap_values, list):
        sv = shap_values[target_class]
    else:
        sv = shap_values

    # sv may be (n_eval_used, n_features) -- enforce 2-D shape.
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # shap >= 0.45 may return (n_eval, n_features, n_classes) as single ndarray
        sv = sv[..., target_class]

    mean_abs_shap = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs_shap)  # descending
    ranked = [(feature_names[i], float(mean_abs_shap[i])) for i in order]

    return {
        'shap_values':     sv,
        'mean_abs_shap':   mean_abs_shap,
        'ranked_features': ranked,
        'n_eval_used':     int(len(X_eval_used)),
        'n_background':    int(n_background),
    }
