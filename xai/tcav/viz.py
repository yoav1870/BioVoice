"""TCAV emergence heatmap visualization.

Renders CAV accuracy as an annotated heatmap showing at which RawNet2 layer
each acoustic concept becomes linearly separable. This is a key paper figure.

Uses seaborn.heatmap with fallback to matplotlib.imshow if seaborn not installed.
Headless backend (Agg) is set at module level for shenkar server operation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


def plot_emergence_heatmap(accuracy_matrix: np.ndarray,
                           layer_names: List[str],
                           concept_names: List[str],
                           output_path: str,
                           title: str = 'Layer-Concept Emergence Heatmap\n(CAV Accuracy -- chance=0.5)') -> None:
    """
    Render CAV accuracy heatmap showing where each concept becomes linearly separable.

    accuracy_matrix: shape (n_layers, n_concepts), values in [0, 1]
    layer_names:     list of friendly layer names (y-axis labels)
    concept_names:   list of concept names (x-axis labels)
    output_path:     path to save PNG (creates parent dirs if needed)
    title:           figure title
    """
    try:
        import seaborn as sns
        _has_seaborn = True
    except ImportError:
        _has_seaborn = False

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(len(concept_names) * 1.5, len(layer_names) * 1.2))

    if _has_seaborn:
        sns.heatmap(
            accuracy_matrix,
            annot=True, fmt='.2f',
            vmin=0.4, vmax=1.0,
            cmap='YlOrRd',
            xticklabels=concept_names,
            yticklabels=layer_names,
            ax=ax,
            linewidths=0.5,
            cbar_kws={'label': 'CAV Accuracy'},
        )
    else:
        # Fallback: matplotlib imshow with manual annotation
        im = ax.imshow(accuracy_matrix, vmin=0.4, vmax=1.0, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(concept_names)))
        ax.set_xticklabels(concept_names)
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names)
        for i in range(len(layer_names)):
            for j in range(len(concept_names)):
                ax.text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=10)
        plt.colorbar(im, ax=ax, label='CAV Accuracy')

    ax.set_title(title)
    ax.set_xlabel('Acoustic Concept')
    ax.set_ylabel('RawNet2 Layer')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
