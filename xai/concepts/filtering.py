"""Threshold-based concept filtering from pre-extracted features."""
import csv
import os
import yaml
from xai.concepts.manifest import write_manifest


def load_features_csv(features_csv_path: str) -> list:
    """Load bonafide_features.csv into a list of dicts with float-cast measure values."""
    rows = []
    with open(features_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ['mean_hnr', 'f0_std', 'spectral_flux_var', 'energy_envelope_var']:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def filter_by_threshold(rows: list, measure: str, threshold: float, direction: str) -> list:
    """Filter rows by threshold on a given measure.

    Args:
        rows: List of dicts with measure values (floats).
        measure: Column name to filter on (e.g., 'mean_hnr').
        threshold: Numeric threshold value.
        direction: 'below' (keep rows < threshold) or 'above' (keep rows > threshold).

    Returns:
        Filtered list of dicts, sorted by audio_id for determinism.
    """
    if direction == 'below':
        selected = [r for r in rows if r[measure] < threshold]
    elif direction == 'above':
        selected = [r for r in rows if r[measure] > threshold]
    else:
        raise ValueError(f"Unknown direction: {direction}. Must be 'below' or 'above'.")
    selected.sort(key=lambda r: r['audio_id'])
    return selected


def build_all_concept_sets(features_csv_path: str, concepts_yaml_path: str, output_dir: str) -> dict:
    """Apply all concept thresholds from config to features CSV, write per-concept manifests.

    Returns dict mapping concept_name -> {count, speakers, manifest_path, status}.
    """
    with open(concepts_yaml_path) as f:
        config = yaml.safe_load(f)

    rows = load_features_csv(features_csv_path)
    results = {}

    for concept_name, concept_cfg in config['concepts'].items():
        measure = concept_cfg['measure']
        threshold = concept_cfg['threshold']
        direction = concept_cfg['direction']
        min_clips = concept_cfg['min_clips']
        min_speakers = concept_cfg['min_speakers']

        selected = filter_by_threshold(rows, measure, threshold, direction)
        speakers = set(r['speaker_id'] for r in selected)

        concept_dir = os.path.join(output_dir, concept_name)
        manifest_path = os.path.join(concept_dir, 'manifest.csv')
        columns = ['audio_id', 'speaker_id', measure]
        manifest_rows = [
            {'audio_id': r['audio_id'], 'speaker_id': r['speaker_id'], measure: r[measure]}
            for r in selected
        ]
        write_manifest(manifest_rows, manifest_path, columns)

        status = 'OK'
        if len(selected) < min_clips:
            status = f'WARNING: {len(selected)} clips < min {min_clips}'
        if len(speakers) < min_speakers:
            status = f'WARNING: {len(speakers)} speakers < min {min_speakers}'

        results[concept_name] = {
            'count': len(selected),
            'speakers': len(speakers),
            'manifest_path': manifest_path,
            'status': status,
        }
        print(f"  {concept_name}: {len(selected)} clips, {len(speakers)} speakers [{status}]")

    return results
