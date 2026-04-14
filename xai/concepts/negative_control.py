"""Negative control concept generation for TCAV statistical validation.

Following Kim et al. 2018 TCAV methodology: random concepts from the same
data distribution serve as the baseline for significance testing. Random
concepts should produce non-significant TCAV scores (p > 0.05 after
Bonferroni correction).
"""
import numpy as np
import os
from xai.concepts.manifest import write_manifest


def generate_negative_control(
    features_rows: list,
    n_clips: int = 200,
    seed: int = 42,
    exclude_audio_ids: set = None,
) -> list:
    """Select random bonafide clips as negative control concept.

    No acoustic threshold applied -- pure random selection from same
    bonafide pool.

    Args:
        features_rows: List of dicts from load_features_csv.
        n_clips: Number of clips to select.
        seed: Random seed for deterministic selection.
        exclude_audio_ids: Optional set of audio_ids to exclude
            (e.g., clips already in concept sets).

    Returns:
        List of selected row dicts.
    """
    rng = np.random.RandomState(seed)

    candidates = features_rows
    if exclude_audio_ids:
        candidates = [r for r in candidates if r['audio_id'] not in exclude_audio_ids]

    n_select = min(n_clips, len(candidates))
    indices = rng.choice(len(candidates), size=n_select, replace=False)
    indices.sort()  # Deterministic ordering
    selected = [candidates[i] for i in indices]
    selected.sort(key=lambda r: r['audio_id'])  # Final sort by audio_id

    return selected


def build_negative_control(features_csv_path: str, concepts_yaml_path: str,
                           output_dir: str, concept_audio_ids: set = None) -> dict:
    """Build negative control concept set from config and features.

    Args:
        features_csv_path: Path to bonafide_features.csv.
        concepts_yaml_path: Path to concepts.yaml (for n_clips and seed).
        output_dir: Base output directory for concept sets.
        concept_audio_ids: Set of audio_ids already assigned to real concepts.

    Returns:
        Dict with count, speakers, manifest_path.
    """
    import yaml
    from xai.concepts.filtering import load_features_csv

    with open(concepts_yaml_path) as f:
        config = yaml.safe_load(f)

    nc_config = config['negative_control']
    rows = load_features_csv(features_csv_path)

    selected = generate_negative_control(
        rows,
        n_clips=nc_config['n_clips'],
        seed=nc_config['seed'],
        exclude_audio_ids=concept_audio_ids,
    )

    speakers = set(r['speaker_id'] for r in selected)
    concept_dir = os.path.join(output_dir, 'negative_control')
    manifest_path = os.path.join(concept_dir, 'manifest.csv')
    columns = ['audio_id', 'speaker_id']
    manifest_rows = [{'audio_id': r['audio_id'], 'speaker_id': r['speaker_id']} for r in selected]
    write_manifest(manifest_rows, manifest_path, columns)

    print(f"  negative_control: {len(selected)} clips, {len(speakers)} speakers [OK]")

    return {
        'count': len(selected),
        'speakers': len(speakers),
        'manifest_path': manifest_path,
    }
