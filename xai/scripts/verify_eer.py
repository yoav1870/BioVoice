#!/usr/bin/env python3
"""End-to-end EER verification script for RawNet2 on ASVspoof dev partition.

Usage:
    cd /home/SpeakerRec/BioVoice
    .venv/bin/python xai/scripts/verify_eer.py [--max-samples N]
"""

import argparse
import datetime
import os
import sys

# Add xai/ to path for internal imports
xai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, xai_root)

import yaml
import torch
import numpy as np

from models.loader import load_rawnet2
from evaluation.inference import run_inference
from evaluation.eer import compute_eer


def main():
    parser = argparse.ArgumentParser(description="Verify RawNet2 EER on ASVspoof")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to process (None=all). Use for quick testing."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to experiment.yaml (default: xai/config/experiment.yaml)"
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config or os.path.join(xai_root, "config", "experiment.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override max_samples if provided via CLI
    if args.max_samples is not None:
        config["evaluation"]["max_samples"] = args.max_samples

    # Set seed for reproducibility
    seed = config["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = config["experiment"]["device"]
    partition = config["evaluation"]["partition"]
    model_name = config["model"]["name"]
    weights_basename = os.path.basename(config["model"]["weights_path"])

    print("=" * 50)
    print("=== RawNet2 EER Verification ===")
    print("=" * 50)

    # Load model
    print("\n[1/3] Loading model...")
    model = load_rawnet2(
        config_path=config["model"]["config_path"],
        weights_path=config["model"]["weights_path"],
        device=device,
    )

    # Run inference
    print("\n[2/3] Running inference...")
    labels, scores = run_inference(model, config, device)

    # Compute EER
    print("\n[3/3] Computing EER...")
    eer, threshold = compute_eer(labels, scores)

    n_bonafide = int(np.sum(labels == 1))
    n_spoof = int(np.sum(labels == 0))

    print()
    print("=" * 50)
    print("=== RawNet2 EER Verification ===")
    print("  Dataset: ASVspoof5")
    print("  Partition: " + partition)
    print("  Samples: {} bonafide, {} spoof".format(n_bonafide, n_spoof))
    print("  Model: {} ({})".format(model_name, weights_basename))
    print("  EER: {:.2f}%".format(eer * 100))
    print("  Threshold: {:.4f}".format(threshold))
    print("=== VERIFICATION COMPLETE ===")
    print("=" * 50)

    # Save results
    results_dir = os.path.join(xai_root, config["output"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_file = os.path.join(
        results_dir, "eer_verification_{}.yaml".format(timestamp)
    )

    results = {
        "experiment": config["experiment"]["name"],
        "timestamp": timestamp,
        "dataset": "ASVspoof5",
        "partition": partition,
        "model": model_name,
        "weights": weights_basename,
        "n_bonafide": n_bonafide,
        "n_spoof": n_spoof,
        "n_total": n_bonafide + n_spoof,
        "eer": float(eer),
        "eer_percent": round(float(eer) * 100, 4),
        "threshold": float(threshold),
        "max_samples": config["evaluation"].get("max_samples"),
        "seed": seed,
    }

    with open(results_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    print("\nResults saved to: " + results_file)
    return eer


if __name__ == "__main__":
    main()
