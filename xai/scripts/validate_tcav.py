#!/usr/bin/env python3
"""Validate TCAV pipeline results: negative control, significance, result completeness."""
import sys
import json
from pathlib import Path

XAI_ROOT = Path(__file__).parent.parent


def validate_tcav(results_path: str = None):
    if results_path is None:
        results_path = str(XAI_ROOT / 'results' / 'tcav' / 'scores' / 'tcav_results.json')

    if not Path(results_path).exists():
        print(f"FAIL: Results file not found: {results_path}")
        return 1

    with open(results_path, 'r') as f:
        results = json.load(f)

    errors = []
    warnings = []

    # Check 1: Results completeness
    expected_concepts = ['breathiness', 'pitch_monotony', 'spectral_smoothness',
                         'temporal_regularity', 'negative_control']
    expected_layers = ['sinc_conv', 'resblock_g1', 'resblock_g2', 'pre_gru', 'post_gru']

    for concept in expected_concepts:
        if concept not in results.get('tcav_scores', {}):
            errors.append(f"FAIL: Missing TCAV scores for concept '{concept}'")
        for layer in expected_layers:
            key = f"{concept}_{layer}"
            if key not in results.get('significance', {}):
                errors.append(f"FAIL: Missing significance result for {key}")

    # Check 2: Negative control MUST NOT be significant (pipeline validity -- TCAV-06)
    for layer in expected_layers:
        key = f"negative_control_{layer}"
        if key in results.get('significance', {}):
            sig = results['significance'][key]
            if sig['significant']:
                errors.append(
                    f"FAIL: negative_control is SIGNIFICANT at {layer} "
                    f"(p_corrected={sig['pval_corrected']:.4f}) -- PIPELINE INVALID"
                )

    # Check 3: At least one real concept should be significant at some layer
    any_significant = False
    for concept in expected_concepts:
        if concept == 'negative_control':
            continue
        for layer in expected_layers:
            key = f"{concept}_{layer}"
            if key in results.get('significance', {}):
                if results['significance'][key]['significant']:
                    any_significant = True
                    break
        if any_significant:
            break

    if not any_significant:
        warnings.append(
            "WARN: No real concepts are significant at any layer -- "
            "findings may be weak but pipeline is valid"
        )

    # Check 4: TCAV scores are in valid range [0, 1]
    for concept in expected_concepts:
        if concept in results.get('tcav_scores', {}):
            for layer in expected_layers:
                if layer in results['tcav_scores'][concept]:
                    for cls in ['bonafide', 'spoof']:
                        score = results['tcav_scores'][concept][layer].get(cls)
                        if score is not None and not (0.0 <= score <= 1.0):
                            errors.append(
                                f"FAIL: {concept}/{layer}/{cls} TCAV score {score} out of [0,1]"
                            )

    # Check 5: Heatmap file exists
    heatmap_path = XAI_ROOT / 'results' / 'tcav' / 'figures' / 'emergence_heatmap.png'
    if not heatmap_path.exists():
        errors.append(f"FAIL: Emergence heatmap not found at {heatmap_path}")

    # Print results
    print("=== TCAV Validation ===")
    if errors:
        for e in errors:
            print(e)
    if warnings:
        for w in warnings:
            print(w)
    if not errors:
        print("All validation checks PASSED.")
        if warnings:
            print(f"({len(warnings)} warning(s))")

    return 1 if errors else 0


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(validate_tcav(path))
