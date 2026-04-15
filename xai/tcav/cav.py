"""CAV (Concept Activation Vector) training for TCAV pipeline.

Trains binary linear probes (CAVs) separating concept activations from random
baseline activations using SGDClassifier with hinge loss. Returns unit-normalized
weight vector, test accuracy, and fitted StandardScaler for reuse on test data.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any


def train_cav(concept_acts: np.ndarray, random_acts: np.ndarray,
              random_state: int = 42) -> Dict[str, Any]:
    """Train binary linear probe (CAV) separating concept from random activations.

    Implements Kim et al. 2018 TCAV: linear probe trained to separate concept
    clips from random baseline clips. The CAV direction (weight vector) is used
    downstream for concept sensitivity scoring.

    Implementation notes:
    - Downsamples to equal class sizes to avoid class imbalance bias (03-RESEARCH.md Pitfall 6)
    - StandardScaler fitted on training data and returned for reuse on test activations
    - Near-50% accuracy is valid (per Kim 2018) -- report it, do not filter

    Args:
        concept_acts: (n_concept, d) pooled activations from concept clips.
        random_acts:  (n_random, d) pooled activations from random/negative clips.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with:
            'cav': np.ndarray of shape (d,) -- unit-normalized weight vector
            'accuracy': float -- test set accuracy
            'scaler': StandardScaler -- fitted scaler (must reuse on test activations)
    """
    # Downsample to equal class sizes (03-RESEARCH.md Pitfall 6)
    n = min(len(concept_acts), len(random_acts))
    rng = np.random.RandomState(random_state)
    c_idx = rng.choice(len(concept_acts), n, replace=False)
    r_idx = rng.choice(len(random_acts), n, replace=False)
    X = np.vstack([concept_acts[c_idx], random_acts[r_idx]])
    y = np.array([1] * n + [0] * n)

    # Standardize -- CRITICAL: scaler saved and reused on test activations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf = SGDClassifier(loss='hinge', alpha=0.01, max_iter=1000,
                        tol=1e-3, random_state=random_state)
    clf.fit(X_train, y_train)

    cav = clf.coef_[0]
    cav = cav / (np.linalg.norm(cav) + 1e-8)  # unit vector
    return {
        'cav': cav,
        'accuracy': float(clf.score(X_test, y_test)),
        'scaler': scaler,
    }
