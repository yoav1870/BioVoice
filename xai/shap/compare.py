"""SHAP <-> TCAV comparison (Phase 5, SHAP-04).

Implements the user-locked D-01 concept->features mapping + D-02 Spearman rho
comparison + D-03 classification (confirms / challenged / inconclusive), with
a schema validator for the Phase 4 per_system_results.json input (Pitfall 7
guard against schema drift).

All functions are pure: no file I/O except load_tcav_results's single
read, no global state, NaN-safe (float('nan') instead of raise).
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr


# Required top-level keys in per_system_results.json (Phase 4 schema,
# 04-02-SUMMARY, 04-VERIFICATION.md). If the Phase 4 schema ever changes,
# update this tuple in sync -- the schema validator will catch the drift.
_REQUIRED_TCAV_KEYS: Tuple[str, ...] = (
    'systems_analyzed',
    'per_system_scores',
    'concept_signatures',
    'transferability',
    'n_significant',
    'fdr_method',
    'n_tests_total',
)


def load_tcav_results(per_system_json: str) -> dict:
    """Load and schema-validate per_system_results.json.

    Raises KeyError listing any missing required keys (Pitfall 7 schema-drift
    guard). Does NOT validate nested structure -- that is the caller's
    responsibility (mean_tcav_score_per_concept guards against missing
    per-system / per-concept / per-layer branches via .get() with defaults).
    """
    data = json.loads(Path(per_system_json).read_text())
    missing = [k for k in _REQUIRED_TCAV_KEYS if k not in data]
    if missing:
        raise KeyError(
            f"per_system_results.json missing required keys: {missing}. "
            f"Phase 4 schema drift suspected; aborting comparison."
        )
    return data


def mean_tcav_score_per_concept(tcav_results: dict,
                                concepts: List[str]) -> Dict[str, float]:
    """Mean TCAV score across (system, layer) pairs where significance=True.

    Concept with zero significant (system, layer) pairs -> 0.5 (null-hypothesis
    TCAV score; matches Phase 3/4 convention where 0.5 is "no directional
    sensitivity").
    """
    out: Dict[str, float] = {}
    for concept in concepts:
        scores = []
        for sys_id, sys_data in tcav_results.get('per_system_scores', {}).items():
            c_data = sys_data.get(concept, {})
            for layer, layer_data in c_data.items():
                if layer_data.get('significant', False):
                    scores.append(layer_data.get('mean_score', 0.5))
        out[concept] = float(np.mean(scores)) if scores else 0.5
    return out


def compare_shap_tcav(mean_abs_shap_by_name: Dict[str, float],
                      tcav_results: dict,
                      concept_to_features: Dict[str, List[str]],
                      rho_confirm: float = 0.7,
                      rho_challenge: float = 0.3) -> Dict:
    """D-02 Spearman rho comparison + D-03 status classification.

    Per-concept rho (operationalization locked to D-02, see 05-RESEARCH.md Open Q 1):
      Within each concept's mapped feature list, treat list order (position in
      concept_to_features[concept]) as the a-priori expected importance rank,
      and the observed rank is the SHAP rank (argsort of -mean_abs_shap for
      mapped features). rho measures whether SHAP agrees with the researcher's
      a-priori ordering.

    Overall rho:
      Rank concepts by mean TCAV significance (across significant system x layer
      pairs) versus rank concepts by their top mapped SHAP feature importance.
      This is the primary scientific number and drives `comparison_status`.

    D-03: status is 'confirms' | 'challenged' | 'inconclusive'. Never raises.
    """
    concept_tcav = mean_tcav_score_per_concept(
        tcav_results, list(concept_to_features)
    )

    per_concept: Dict[str, Dict] = {}
    tcav_per_concept: List[float] = []
    top_shap_per_concept: List[float] = []

    for concept, features in concept_to_features.items():
        mapped = [f for f in features if f in mean_abs_shap_by_name]
        if len(mapped) < 2:
            per_concept[concept] = {
                'rho':      float('nan'),
                'pval':     float('nan'),
                'n_mapped': int(len(mapped)),
                'note':     'too_few_mapped_features',
            }
            # Still record this concept's TCAV score for overall rho if it
            # happens to have exactly 1 mapped feature -- use that single
            # feature's SHAP as "top SHAP".
            if len(mapped) == 1:
                tcav_per_concept.append(concept_tcav[concept])
                top_shap_per_concept.append(
                    float(mean_abs_shap_by_name[mapped[0]])
                )
            continue

        expected_rank = list(range(len(mapped)))
        shap_vals = np.array([mean_abs_shap_by_name[f] for f in mapped])
        shap_rank = np.argsort(-shap_vals).argsort()
        rho, pval = spearmanr(expected_rank, shap_rank)

        per_concept[concept] = {
            'rho':      float(rho)  if not np.isnan(rho)  else float('nan'),
            'pval':     float(pval) if not np.isnan(pval) else float('nan'),
            'n_mapped': int(len(mapped)),
        }
        tcav_per_concept.append(concept_tcav[concept])
        top_shap_per_concept.append(float(np.max(shap_vals)))

    # Overall rho: rank across concepts (need >= 2 for spearmanr).
    if len(tcav_per_concept) >= 2:
        overall_rho, overall_pval = spearmanr(
            tcav_per_concept, top_shap_per_concept
        )
    else:
        overall_rho, overall_pval = float('nan'), float('nan')

    # D-03 classification. Never aborts -- report + continue on 'challenged'.
    if np.isnan(overall_rho):
        status = 'inconclusive'
    elif overall_rho > rho_confirm:
        status = 'confirms'
    elif overall_rho < rho_challenge:
        status = 'challenged'
    else:
        status = 'inconclusive'

    return {
        'per_concept_rho': per_concept,
        'overall_rho': {
            'rho':  float(overall_rho)  if not np.isnan(overall_rho)  else float('nan'),
            'pval': float(overall_pval) if not np.isnan(overall_pval) else float('nan'),
        },
        'comparison_status': status,
    }
