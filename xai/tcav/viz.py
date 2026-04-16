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


def plot_per_system_heatmap(score_matrix, system_names, concept_names,
                            significance_mask, output_path,
                            title='Per-System Concept Attribution'):
    """
    Heatmap of TCAV scores per synthesis system and acoustic concept.

    Rows = synthesis systems (A09-A16), columns = acoustic concepts.
    Cells annotated with score value; significant cells marked with '*'.
    Memory: plt.close() called after save (T-03-07).

    Args:
        score_matrix: (n_systems, n_concepts) array of mean TCAV scores
        system_names: list of system IDs (rows)
        concept_names: list of concept names (columns)
        significance_mask: (n_systems, n_concepts) bool array -- True = significant
        output_path: file path to save PNG
        title: figure title
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        import seaborn as sns
        _has_seaborn = True
    except ImportError:
        _has_seaborn = False

    fig, ax = plt.subplots(figsize=(len(concept_names) * 1.8, max(len(system_names) * 0.8, 4)))
    annot = np.empty_like(score_matrix, dtype=object)
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            val = f'{score_matrix[i, j]:.2f}'
            annot[i, j] = f'{val}*' if significance_mask[i, j] else val

    if _has_seaborn:
        sns.heatmap(score_matrix, annot=annot, fmt='',
                    vmin=0.3, vmax=0.9, cmap='YlOrRd',
                    xticklabels=concept_names, yticklabels=system_names,
                    ax=ax, linewidths=0.5,
                    cbar_kws={'label': 'TCAV Score (spoof class)'})
    else:
        im = ax.imshow(score_matrix, vmin=0.3, vmax=0.9, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(concept_names)))
        ax.set_xticklabels(concept_names, rotation=45, ha='right')
        ax.set_yticks(range(len(system_names)))
        ax.set_yticklabels(system_names)
        for i in range(score_matrix.shape[0]):
            for j in range(score_matrix.shape[1]):
                ax.text(j, i, annot[i, j], ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax, label='TCAV Score (spoof class)')

    ax.set_title(title)
    ax.set_xlabel('Acoustic Concept')
    ax.set_ylabel('Synthesis System')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # memory cleanup (T-03-07)


def plot_transferability(score_matrix, system_names, concept_names, output_path):
    """
    Hierarchically clustered heatmap showing system-concept transferability.

    Rows = synthesis systems, columns = acoustic concepts.
    Ward linkage hierarchical clustering groups similar systems and concepts.
    Memory: plt.close() called after save (T-03-07).

    Args:
        score_matrix: (n_systems, n_concepts) array of mean TCAV scores
        system_names: list of system IDs (rows)
        concept_names: list of concept names (columns)
        output_path: file path to save PNG
    """
    import pandas as pd

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        import seaborn as sns
        df = pd.DataFrame(score_matrix, index=system_names, columns=concept_names)
        g = sns.clustermap(df, method='ward', cmap='YlOrRd',
                           annot=True, fmt='.2f', vmin=0.3, vmax=0.9,
                           figsize=(len(concept_names) * 2, len(system_names) * 1.2),
                           linewidths=0.5,
                           cbar_kws={'label': 'Mean TCAV Score'},
                           dendrogram_ratio=(0.15, 0.15))
        g.fig.suptitle('Concept Transferability: System-Concept Clustering', y=1.02)
        g.savefig(output_path, dpi=150, bbox_inches='tight')
    except ImportError:
        # Fallback: plain heatmap if seaborn unavailable
        fig, ax = plt.subplots(figsize=(len(concept_names) * 2, len(system_names) * 1.2))
        im = ax.imshow(score_matrix, vmin=0.3, vmax=0.9, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(concept_names)))
        ax.set_xticklabels(concept_names, rotation=45, ha='right')
        ax.set_yticks(range(len(system_names)))
        ax.set_yticklabels(system_names)
        ax.set_title('Concept Transferability: System-Concept Mean TCAV Scores')
        plt.colorbar(im, ax=ax, label='Mean TCAV Score')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # memory cleanup (T-03-07)
