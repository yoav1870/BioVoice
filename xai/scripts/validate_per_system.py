#!/usr/bin/env python3
"""Validate per-system TCAV results: system counts, FDR correction, negative control, transferability.

Usage:
    cd /home/SpeakerRec/BioVoice
    .venv/bin/python xai/scripts/validate_per_system.py [results_path]
"""
import sys
import json
from pathlib import Path

XAI_ROOT = Path(__file__).parent.parent


def validate_per_system(results_path=None):
    """
    Validate per-system TCAV results JSON.

    Checks:
      1. Required top-level keys present (systems_analyzed, per_system_scores, transferability)
      2. Negative control is non-significant for all systems (pipeline validity)
      3. FDR correction applied (n_significant field present, fdr_method = benjamini_hochberg)
      4. At least one system analyzed (not all excluded)

    Returns 0 on success, 1 on any failure.
    """
    if results_path is None:
        results_path = str(XAI_ROOT / 'results' / 'per_system' / 'scores' / 'per_system_results.json')

    if not Path(results_path).exists():
        print(f"FAIL: Results file not found: {results_path}")
        return 1

    with open(results_path, 'r') as f:
        results = json.load(f)

    errors = []
    warnings = []

    # Check 1: Required keys present
    required_keys = ['systems_analyzed', 'per_system_scores', 'transferability',
                     'n_significant', 'fdr_method', 'n_tests_total']
    for key in required_keys:
        if key not in results:
            errors.append(f"FAIL: Missing required key in results: '{key}'")

    # Check 2: FDR method is BH
    if results.get('fdr_method') != 'benjamini_hochberg':
        errors.append(f"FAIL: Expected fdr_method=benjamini_hochberg, got: {results.get('fdr_method')}")

    # Check 3: At least one system analyzed
    systems = results.get('systems_analyzed', [])
    if len(systems) == 0:
        errors.append("FAIL: No systems were analyzed (all excluded or protocol empty)")

    # Check 4: Negative control non-significant for all systems
    per_system_scores = results.get('per_system_scores', {})
    nc_significant_systems = []
    for sys_id, concept_data in per_system_scores.items():
        if 'negative_control' in concept_data:
            for layer, layer_data in concept_data['negative_control'].items():
                if layer_data.get('significant', False):
                    nc_significant_systems.append(f"{sys_id}/{layer}")

    if nc_significant_systems:
        errors.append(
            f"FAIL: Negative control significant for systems/layers: {nc_significant_systems}. "
            "This indicates pipeline inflation or incorrect FDR correction."
        )

    # Check 5: Transferability classifications present
    transferability = results.get('transferability', {})
    per_system = results.get('per_system_scores', {})
    if per_system and not transferability:
        warnings.append("WARNING: Transferability classifications missing from results")

    for classification in transferability.values():
        if 'classification' not in classification:
            errors.append("FAIL: Transferability entry missing 'classification' key")
            break

    # Report
    for w in warnings:
        print(w)

    if errors:
        for e in errors:
            print(e)
        print(f"\nValidation FAILED ({len(errors)} errors, {len(warnings)} warnings)")
        return 1

    # Summary of findings
    n_systems = len(systems)
    n_significant = results.get('n_significant', 0)
    n_total = results.get('n_tests_total', 0)
    print(f"Systems analyzed: {n_systems} ({', '.join(sorted(systems))})")
    print(f"Significant findings: {n_significant}/{n_total} tests (FDR alpha={results.get('alpha', 0.05)})")
    for concept, info in transferability.items():
        print(f"  {concept}: {info.get('classification', '?')} "
              f"({info.get('n_significant_systems', 0)}/{info.get('total_systems', n_systems)} systems)")
    print(f"\nAll per-system validation checks PASSED ({len(warnings)} warnings)")
    return 0


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(validate_per_system(path))
