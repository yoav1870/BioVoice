"""Pass 1: Stream ASVspoof5 train tars, extract acoustic features for bonafide clips.

This is the expensive step (~60-75 min on shenkar for 18,797 bonafide clips).
Run once to produce bonafide_features.csv, then use build_concepts.py (Pass 2).

Security: T-02-03 basename prevents tar traversal; T-02-04 checks len >= 9.
"""
import argparse
import csv
import io
import os
import tarfile
import time
import yaml
import soundfile as sf

from xai.concepts.measures import (
    compute_mean_hnr,
    compute_f0_std,
    compute_spectral_flux_var,
    compute_energy_envelope_var,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract acoustic features from ASVspoof5 train bonafide clips.",
    )
    parser.add_argument("--config", default="xai/config/experiment.yaml",
        help="Path to experiment.yaml")
    parser.add_argument("--concepts-config", default="xai/config/concepts.yaml",
        help="Path to concepts.yaml")
    parser.add_argument("--output", default="xai/data/features/bonafide_features.csv",
        help="Output CSV path")
    return parser.parse_args()


def _load_bonafide_index(protocol_path: str) -> dict:
    bonafide = {}
    skipped = 0
    verified = False
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if not verified:
                print(f"Protocol verification -- first row: {line.strip()}")
                if len(parts) >= 9:
                    print(f"  parts[0] (speaker_id): {parts[0]}")
                    print(f"  parts[1] (audio_id): {parts[1]}")
                    print(f"  parts[8] (key): {parts[8]}")
                    assert parts[8] in ("bonafide", "spoof"), (
                        f"Column 8 is {parts[8]!r}, expected bonafide/spoof."
                    )
                verified = True
            if len(parts) < 9:
                print(f"  WARNING: Skipping malformed row ({len(parts)} cols)")
                skipped += 1
                continue
            speaker_id = parts[0]
            audio_id = parts[1]
            key = parts[8]  # CRITICAL: bonafide/spoof at index 8, NOT 7
            if key == "bonafide":
                bonafide[audio_id] = speaker_id
    if skipped > 0:
        print(f"  WARNING: Skipped {skipped} malformed rows in protocol")
    return bonafide


def extract_all_bonafide_features(
    train_tars_dir: str,
    bonafide: dict,
    output_csv_path: str,
    sample_rate: int = 16000,
) -> int:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    tar_files = sorted(f for f in os.listdir(train_tars_dir) if f.endswith(".tar"))
    print(f"Tar files found: {len(tar_files)} -- {tar_files}")
    columns = ["audio_id", "speaker_id", "mean_hnr", "f0_std", "spectral_flux_var", "energy_envelope_var"]
    processed = 0
    errors = 0
    start_time = time.time()
    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for tar_name in tar_files:
            tar_path = os.path.join(train_tars_dir, tar_name)
            print(f"Processing {tar_name}...")
            with tarfile.open(tar_path, "r") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".flac"):
                        continue
                    # T-02-03: os.path.basename prevents tar path traversal
                    audio_id = os.path.splitext(os.path.basename(member.name))[0]
                    if audio_id not in bonafide:
                        continue
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            errors += 1
                            continue
                        audio_bytes = f.read()
                        audio, sr = sf.read(io.BytesIO(audio_bytes))
                        if audio.ndim > 1:
                            audio = audio.mean(axis=1)
                        mean_hnr = compute_mean_hnr(audio, sr)
                        f0_std = compute_f0_std(audio, sr)
                        spectral_flux_var = compute_spectral_flux_var(audio, sr)
                        energy_envelope_var = compute_energy_envelope_var(audio, sr)
                        writer.writerow({
                            "audio_id": audio_id,
                            "speaker_id": bonafide[audio_id],
                            "mean_hnr": mean_hnr,
                            "f0_std": f0_std,
                            "spectral_flux_var": spectral_flux_var,
                            "energy_envelope_var": energy_envelope_var,
                        })
                        csv_file.flush()
                        processed += 1
                        if processed % 500 == 0:
                            elapsed = time.time() - start_time
                            rate = processed / elapsed
                            remaining = len(bonafide) - processed
                            eta_min = (remaining / rate) / 60 if rate > 0 else 0
                            print(f"  [{processed}/{len(bonafide)}] {rate:.1f} clips/s  ETA: {eta_min:.0f} min")
                    except Exception as e:
                        print(f"  WARNING: Failed to process {audio_id}: {e}")
                        errors += 1
    elapsed = time.time() - start_time
    print("")
    print(f"Extraction complete: {processed} clips processed, {errors} errors")
    print(f"Total time: {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    print(f"Output: {output_csv_path}")
    return processed


def main():
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_tars_dir = config["data"]["train_tars_dir"]
    train_protocol = config["data"]["train_protocol"]
    sample_rate = config["data"].get("sample_rate", 16000)
    print(f"Config loaded from: {args.config}")
    print(f"Train tars dir: {train_tars_dir}")
    print(f"Train protocol: {train_protocol}")
    print(f"Output CSV: {args.output}")
    print()
    print("Parsing train protocol...")
    bonafide = _load_bonafide_index(train_protocol)
    print(f"Bonafide clips to process: {len(bonafide)}")
    print()
    print("Starting feature extraction (~60-75 min on shenkar)...")
    processed = extract_all_bonafide_features(
        train_tars_dir=train_tars_dir,
        bonafide=bonafide,
        output_csv_path=args.output,
        sample_rate=sample_rate,
    )
    print("")
    print(f"Done. {processed} bonafide clips in {args.output}")


if __name__ == "__main__":
    main()
