"""Quick test runner for TCAV - single speaker (id00012)."""

from __future__ import annotations

from tcav_core.config_test import TEST_CONFIG
from tcav_core.runner import TCAVRunner


def main() -> None:
    """Run TCAV on test dataset (single speaker, reduced samples)."""
    print("\n" + "=" * 70)
    print("RUNNING TCAV TEST - SINGLE SPEAKER (id00012)")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Dataset: {TEST_CONFIG.dataset_csv_path.name}")
    print(f"  Samples: ~10 audio files")
    print(f"  Layer: {TEST_CONFIG.target_layer_keys}")
    print(f"  Concept samples: {TEST_CONFIG.concept_samples} (reduced for speed)")
    print(f"  Random samples: {TEST_CONFIG.random_samples} (reduced for speed)")
    print(f"  CAV cache: {TEST_CONFIG.cav_save_path.name}")
    print(f"  Output tag: {TEST_CONFIG.output_tag}")
    print("\n" + "=" * 70 + "\n")

    runner = TCAVRunner(TEST_CONFIG)
    runner.run()

    print("\n" + "=" * 70)
    print("✅ TEST RUN COMPLETE!")
    print(f"Results saved to: {TEST_CONFIG.output_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
