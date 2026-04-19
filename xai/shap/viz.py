"""FIG-06 (SHAP feature importance) for the SHAP baseline track (Phase 5).

Horizontal bar chart of the top-k features by mean absolute SHAP value, with a
text annotation box listing per-concept Spearman rho and the overall comparison
status (CONTEXT.md D-04).

Headless-backend and memory-leak discipline carried forward from xai/tcav/viz.py:
matplotlib.use('Agg') MUST be set before importing pyplot, and plt.close() MUST
be called after every plt.savefig() (T-03-07 in 05-RESEARCH.md Security Domain).
"""
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # MUST precede pyplot import -- shenkar is headless
import matplotlib.pyplot as plt  # noqa: E402  (intentional ordering)
import numpy as np  # noqa: E402


def plot_shap_importance(ranked_features: List[Tuple[str, float]],
                         per_concept_rho: Dict[str, Dict],
                         overall_rho: float,
                         comparison_status: str,
                         output_path: str,
                         top_k: int = 15,
                         title: str = 'SHAP Feature Importance (top 15)') -> None:
    """FIG-06 horizontal bar chart with Spearman rho annotation box.

    Parameters:
        ranked_features: [(feature_name, mean_abs_shap), ...] sorted descending.
                         Only the first `top_k` are plotted.
        per_concept_rho: {concept_name: {'rho': float, 'pval': float, 'n_mapped': int, ...}}
                         (output of compare.compare_shap_tcav).
        overall_rho:     scalar overall Spearman rho (compare.compare_shap_tcav
                         returns this as a nested dict; the caller unwraps).
        comparison_status: 'confirms' | 'challenged' | 'inconclusive'.
        output_path: file path for the PNG.
        top_k: number of features to plot (default 15 -- matches D-04).
        title: chart title.

    Side effects: writes a PNG (>= 50 KB at dpi=150) and calls plt.close().
    """
    top = ranked_features[:top_k]
    # Reversed so the largest bar is at the top of the chart.
    names  = [n for n, _ in reversed(top)]
    values = [v for _, v in reversed(top)]

    fig, ax = plt.subplots(figsize=(10, 0.45 * top_k + 2))
    ax.barh(names, values, color='#4C72B0')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(title)

    # D-04 annotation box: per-concept rho, overall rho, status.
    lines = ['Spearman rho vs TCAV:']
    for concept, info in per_concept_rho.items():
        rho = info.get('rho')
        if rho is None or (isinstance(rho, float) and np.isnan(rho)):
            rho_str = 'NaN'
        else:
            rho_str = f'{rho:+.2f}'
        lines.append(f'  {concept}: {rho_str}')
    if overall_rho is None or (isinstance(overall_rho, float) and np.isnan(overall_rho)):
        lines.append('Overall rho: NaN')
    else:
        lines.append(f'Overall rho: {overall_rho:+.2f}')
    lines.append(f'Status: {comparison_status}')
    annotation = '\n'.join(lines)

    ax.text(
        0.98, 0.02, annotation,
        transform=ax.transAxes,
        fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='white', edgecolor='gray', alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # T-03-07: prevent matplotlib memory leak
