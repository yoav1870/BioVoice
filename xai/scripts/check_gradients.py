#!/usr/bin/env python
"""Standalone gradient flow verification script for RawNet2.

Loads the pretrained RawNet2 model, runs a forward+backward pass,
and prints a pass/fail table for all 10 target layer groups.

Usage:
    cd /home/SpeakerRec/BioVoice
    .venv/bin/python xai/scripts/check_gradients.py

FOUND-02: Must confirm gradient flow before TCAV analysis.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Ensure xai is importable
_xai_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_xai_root))

from models.loader import load_rawnet2
from models.gradient_check import verify_gradient_flow


def main():
    # Load experiment config
    config_path = _xai_root / "config" / "experiment.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = config["experiment"]["device"]
    model_config = str(_xai_root / config["model"]["config_path"])
    weights = str(_xai_root / config["model"]["weights_path"])

    print("=" * 65)
    print("  RawNet2 GRADIENT CHECK")
    print("=" * 65)
    print()
    print("Config: {}".format(config_path))
    print("Device: {}".format(device))
    print()

    # Load model
    print("Loading model...")
    model = load_rawnet2(model_config, weights, device=device, verbose=False)

    # Create sample input
    sample_input = torch.randn(2, 64600)

    # Run gradient verification
    print("Running forward+backward pass...")
    print()
    results = verify_gradient_flow(model, sample_input, device=device)

    # Print results table
    print("=== RawNet2 GRADIENT CHECK ===")
    print()

    header = "{:<28s} | {:<6s} | {:<12s} | {:<12s} | {:<6s} | {:<10s}".format(
        "Layer", "Status", "Mean |grad|", "Max |grad|", "Params", "Method"
    )
    separator = "-" * len(header)

    print(header)
    print(separator)

    all_pass = True
    fail_layers = []

    for layer_name, info in results.items():
        status = "PASS" if info["has_grad"] else "FAIL"
        if not info["has_grad"]:
            all_pass = False
            fail_layers.append(layer_name)

        mean_str = "{:.2e}".format(info["mean_abs"]) if info["mean_abs"] > 0 else "0.00e+00"
        max_str = "{:.2e}".format(info["max_abs"]) if info["max_abs"] > 0 else "0.00e+00"

        print(
            "{:<28s} | {:<6s} | {:<12s} | {:<12s} | {:<6d} | {:<10s}".format(
                layer_name,
                status,
                mean_str,
                max_str,
                info["num_params"],
                info["method"],
            )
        )

    print()

    if all_pass:
        print("Result: ALL LAYERS PASS -- gradient flow confirmed")
    else:
        print("Result: FAILED -- gradient flow broken at: {}".format(", ".join(fail_layers)))
        print()
        print("Diagnostic suggestions:")
        print("  1. Check for .detach() calls in the model's forward method")
        print("  2. Check for in-place operations (+=, .mul_(), etc.)")
        print("  3. Check for torch.no_grad() wrapping the forward pass")
        print("  4. Verify the model is in train() mode during gradient check")
        print()
        for fl in fail_layers:
            info = results[fl]
            if info["details"]:
                print("  {}: {}".format(fl, info["details"]))

    print("=== GRADIENT CHECK COMPLETE ===")

    # Save results to YAML
    results_dir = _xai_root / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_file = results_dir / "gradient_check.yaml"

    # Build YAML-safe results dict
    yaml_results = {}
    for layer_name, info in results.items():
        yaml_results[layer_name] = {
            "has_grad": bool(info["has_grad"]),
            "mean_abs": float(info["mean_abs"]),
            "max_abs": float(info["max_abs"]),
            "num_params": int(info["num_params"]),
            "method": str(info["method"]),
            "details": str(info["details"]),
        }

    output = {
        "timestamp": timestamp,
        "device": device,
        "model_weights": os.path.basename(weights),
        "sample_shape": [2, 64600],
        "backward_target": "output[:, 1].sum() (bonafide class logit)",
        "all_pass": all_pass,
        "layers": yaml_results,
    }

    with open(results_file, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print()
    print("Results saved: {}".format(results_file))

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
