"""Pass 2: Apply concept thresholds to bonafide_features.csv and build concept manifests.

This is fast (seconds) and can be re-run with different thresholds in concepts.yaml
without re-extracting features (Pass 1 extract_features.py runs once).

Usage:
    python -m xai.scripts.build_concepts --features xai/data/features/bonafide_features.csv
"""
import argparse
import csv
import os

from xai.concepts.filtering import build_all_concept_sets
from xai.concepts.negative_control import build_negative_control


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Apply concept thresholds to features CSV and build concept manifests.",
    )
    parser.add_argument(
        "--features",
        default="xai/data/features/bonafide_features.csv",
        help="Path to bonafide_features.csv produced by extract_features.py",
    )
    parser.add_argument(
        "--concepts-config",
        default="xai/config/concepts.yaml",
        help="Path to concepts.yaml with thresholds",
    )
    parser.add_argument(
        "--output",
        default="xai/data/concepts/",
        help="Output directory for concept manifests",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    print(f"Building concept sets from: {args.features}")
    print(f"Config: {args.concepts_config}")
    print(f"Output: {args.output}")
    print()

    # Pass 2a: Apply thresholds -- produces per-concept manifests
    print("Applying concept thresholds...")
    results = build_all_concept_sets(
        features_csv_path=args.features,
        concepts_yaml_path=args.concepts_config,
        output_dir=args.output,
    )

    # Collect all audio_ids assigned to real concepts (for negative control exclusion)
    all_concept_ids = set()
    for concept_name, info in results.items():
        manifest_path = info["manifest_path"]
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_concept_ids.add(row["audio_id"])

    # Pass 2b: Build negative control (random selection, excluding concept clips)
    print("Building negative control...")
    nc_result = build_negative_control(
        features_csv_path=args.features,
        concepts_yaml_path=args.concepts_config,
        output_dir=args.output,
        concept_audio_ids=all_concept_ids,
    )
    results["negative_control"] = nc_result

    # Print summary table
    print()
    header = "{:<25} {:>8} {:>10}   {}".format("Concept", "Clips", "Speakers", "Status")
    print("-" * 70)
    print(header)
    print("-" * 70)
    for concept_name, info in results.items():
        count = info["count"]
        speakers = info["speakers"]
        status = info.get("status", "OK")
        row_str = "{:<25} {:>8} {:>10}   {}".format(concept_name, count, speakers, status)
        print(row_str)
    print("-" * 70)
    total = sum(v["count"] for v in results.values())
    print(f"Total concept clips (incl. negative_control): {total}")


if __name__ == "__main__":
    main()
