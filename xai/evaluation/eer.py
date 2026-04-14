"""Equal Error Rate (EER) computation."""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def compute_eer(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Compute Equal Error Rate from binary labels and continuous scores.

    Args:
        labels: Binary array (1=bonafide/target, 0=spoof/nontarget).
        scores: Continuous score array (higher = more likely bonafide).

    Returns:
        (eer, threshold): EER value (0.0 to 1.0) and the threshold at the EER point.

    Raises:
        ValueError: If inputs are invalid.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    # Input validation
    if labels.shape != scores.shape:
        raise ValueError(
            f"labels and scores must have the same shape: "
            f"{labels.shape} vs {scores.shape}"
        )
    if len(labels) == 0:
        raise ValueError("labels and scores must not be empty")

    unique_labels = set(np.unique(labels))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"labels must be binary (0 or 1), got unique values: {unique_labels}"
        )
    if 0 not in unique_labels or 1 not in unique_labels:
        raise ValueError(
            "labels must contain both classes (0 and 1)"
        )

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    # Find EER: the point where FPR == 1 - TPR (i.e., FPR == FNR)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = float(interp1d(fpr, thresholds)(eer))

    return float(eer), threshold
