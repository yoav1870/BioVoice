"""Validate concept set integrity: counts, speakers, and disjointness from dev/eval.

Checks:
  1. Each concept has >= min_clips clips (from concepts.yaml).
  2. Each concept has >= min_speakers distinct speakers.
  3. No speaker_id starts with D_ or E_ (train-only, disjointness check).
  4. No audio_id appears in dev or eval protocol (utterance disjointness).
"""
import argparse
import csv
import os
import sys
import yaml

from xai.concepts.manifest import read_manifest


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Validate concept set integrity (counts, speakers, disjointness).",
    )
    parser.add_argument(
        "--concepts-dir",
        default="xai/data/concepts/",
        help="Directory containing concept manifest subdirectories",
    )
    parser.add_argument(
        "--config",
        default="xai/config/experiment.yaml",
        help="Path to experiment.yaml (for dev/eval protocol paths)",
    )
    parser.add_argument(
        "--concepts-config",
        default="xai/config/concepts.yaml",
        help="Path to concepts.yaml (for min_clips and min_speakers thresholds)",
    )
    return parser.parse_args()


def _load_protocol_ids(protocol_path: str):
    """Load all audio_ids and speaker_ids from a protocol TSV file."""
    audio_ids = set()
    speaker_ids = set()
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            speaker_ids.add(parts[0])
            audio_ids.add(parts[1])
    return audio_ids, speaker_ids


def validate_disjointness(
    concept_rows: list,
    dev_audio_ids: set,
    eval_audio_ids: set,
) -> dict:
    """Check that no concept clip appears in dev or eval protocols.

    Also checks that no speaker_id starts with D_ or E_ (wrong partition).

    Returns dict with:
      - speaker_prefix_ok: bool (all speakers start with T_)
      - utterance_disjoint: bool (no audio_id in dev/eval)
      - bad_speakers: list of speaker_ids with wrong prefix
      - bad_audio_ids: list of audio_ids found in dev/eval
    """
    bad_speakers = []
    bad_audio_ids = []
    combined_held_out = dev_audio_ids | eval_audio_ids

    for row in concept_rows:
        speaker_id = row.get("speaker_id", "")
        audio_id = row.get("audio_id", "")
        if not speaker_id.startswith("T_"):
            bad_speakers.append(speaker_id)
        if audio_id in combined_held_out:
            bad_audio_ids.append(audio_id)

    return {
        "speaker_prefix_ok": len(bad_speakers) == 0,
        "utterance_disjoint": len(bad_audio_ids) == 0,
        "bad_speakers": bad_speakers,
        "bad_audio_ids": bad_audio_ids,
    }


def main():
    args = _parse_args()

    with open(args.config) as f:
        exp_config = yaml.safe_load(f)
    with open(args.concepts_config) as f:
        concept_config = yaml.safe_load(f)

    dev_protocol = exp_config["data"]["dev_protocol"]
    eval_protocol = exp_config["data"]["eval_protocol"]

    print(f"Loading dev protocol from: {dev_protocol}")
    dev_audio_ids, _ = _load_protocol_ids(dev_protocol)
    print(f"  {len(dev_audio_ids)} dev audio IDs loaded")

    print(f"Loading eval protocol from: {eval_protocol}")
    eval_audio_ids, _ = _load_protocol_ids(eval_protocol)
    print(f"  {len(eval_audio_ids)} eval audio IDs loaded")

    print(f"Scanning concept manifests in: {args.concepts_dir}")
    print()

    concept_dirs = sorted(
        d for d in os.listdir(args.concepts_dir)
        if os.path.isdir(os.path.join(args.concepts_dir, d))
    ) if os.path.isdir(args.concepts_dir) else []

    if not concept_dirs:
        print(f"ERROR: No concept subdirectories found in {args.concepts_dir}")
        sys.exit(1)

    all_passed = True
    print("-" * 80)
    print("{:<25} {:>7} {:>10} {:>12} {:>14}".format(
        "Concept", "Clips", "Speakers", "Prefix OK", "Utterance OK"
    ))
    print("-" * 80)

    for concept_name in concept_dirs:
        manifest_path = os.path.join(args.concepts_dir, concept_name, "manifest.csv")
        if not os.path.isfile(manifest_path):
            print(f"{concept_name:<25} MISSING manifest.csv")
            all_passed = False
            continue

        rows = read_manifest(manifest_path)
        n_clips = len(rows)
        speakers = set(r.get("speaker_id", "") for r in rows)
        n_speakers = len(speakers)

        concept_cfg = concept_config.get("concepts", {}).get(concept_name, {})
        min_clips = concept_cfg.get("min_clips", 0)
        min_speakers = concept_cfg.get("min_speakers", 0)

        dis = validate_disjointness(rows, dev_audio_ids, eval_audio_ids)

        clip_ok = n_clips >= min_clips if min_clips > 0 else True
        spk_ok = n_speakers >= min_speakers if min_speakers > 0 else True
        prefix_ok = dis["speaker_prefix_ok"]
        utterance_ok = dis["utterance_disjoint"]
        passed = clip_ok and spk_ok and prefix_ok and utterance_ok
        if not passed:
            all_passed = False

        bad_spk_count = len(dis["bad_speakers"])
        bad_aud_count = len(dis["bad_audio_ids"])
        prefix_str = "OK" if prefix_ok else f"FAIL({bad_spk_count} bad)"
        utterance_str = "OK" if utterance_ok else f"FAIL({bad_aud_count} overlap)"
        clip_str = str(n_clips) + ("" if clip_ok else f" (<{min_clips})")
        spk_str = str(n_speakers) + ("" if spk_ok else f" (<{min_speakers})")

        print("{:<25} {:>7} {:>10} {:>12} {:>14}".format(
            concept_name, clip_str, spk_str, prefix_str, utterance_str
        ))

    print("-" * 80)

    if all_passed:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED -- see details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
