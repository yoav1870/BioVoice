#!/usr/bin/env python3
"""
Build a TCAV-ready ASVspoof5 subset for spoof-vs-bonafide analysis.

Design:
- 6 speakers
- 16 bonafide samples per speaker
- 4 spoof samples per system A01-A04 per speaker

Totals:
- bonafide: 96
- spoof: 96
- total: 192
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/home/SpeakerRec/BioVoice")

MANIFEST = (
    PROJECT_ROOT
    / "data/datasets/ASVspoof5_tars/ASVspoof5_protocols/gender_50spk_2000perclass_A01A08_outputs/male/selected_utterances_plan.csv"
)

AUDIO_TARS_ROOT = (
    PROJECT_ROOT / "data/datasets/ASVspoof5_tars/ASVspoof5_audio_train_tars"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data/datasets/ASVspoof5_tars/analysis_subset_a01_a04_balanced_binary"
)
OUT_AUDIO_DIR = OUT_DIR / "audio"
OUT_CSV = OUT_DIR / "tcav_analysis_subset_spoof.csv"

NUM_SPEAKERS = 6
BONAFIDE_PER_SPK = 16
SPOOF_PER_SYSTEM_PER_SPK = 4
REQUIRED_SYSTEMS = ["A01", "A02", "A03", "A04"]


def exact_extract_from_tars(utt_id: str, target_path: Path, tar_files: list[Path]) -> bool:
    for tar_file in tar_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"tar -xf {tar_file} -C {tmpdir} 'flac_T/{utt_id}.flac' 2>/dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, check=False)
            extracted = Path(tmpdir) / "flac_T" / f"{utt_id}.flac"
            if extracted.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(extracted), str(target_path))
                return True
    return False


def main() -> None:
    print("=" * 80)
    print("BUILD ANALYSIS SUBSET")
    print("=" * 80)

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    print(f"Loaded manifest: {len(df)} rows")

    df = df[
        ((df["label"] == "bonafide") & (df["system_id"] == "bonafide"))
        | ((df["label"] == "spoof") & (df["system_id"].isin(REQUIRED_SYSTEMS)))
    ].copy()

    eligible = []
    for speaker_id, group in df.groupby("speaker_id"):
        bon_n = int((group["label"] == "bonafide").sum())
        sys_counts = {
            sys_id: int(
                ((group["label"] == "spoof") & (group["system_id"] == sys_id)).sum()
            )
            for sys_id in REQUIRED_SYSTEMS
        }
        if bon_n >= BONAFIDE_PER_SPK and all(
            sys_counts[sys_id] >= SPOOF_PER_SYSTEM_PER_SPK
            for sys_id in REQUIRED_SYSTEMS
        ):
            eligible.append(
                {
                    "speaker_id": speaker_id,
                    "bonafide_n": bon_n,
                    **{f"{sys_id}_n": sys_counts[sys_id] for sys_id in REQUIRED_SYSTEMS},
                    "total_rows": len(group),
                }
            )

    eligible_df = pd.DataFrame(eligible).sort_values(
        ["bonafide_n", "total_rows"], ascending=False
    )

    if len(eligible_df) < NUM_SPEAKERS:
        raise RuntimeError(
            f"Not enough eligible speakers. Need {NUM_SPEAKERS}, found {len(eligible_df)}"
        )

    selected_speakers = eligible_df["speaker_id"].head(NUM_SPEAKERS).tolist()
    print(f"Selected speakers ({len(selected_speakers)}): {selected_speakers}")

    rows = []
    for speaker_id in selected_speakers:
        speaker_df = df[df["speaker_id"] == speaker_id].copy()

        bon = (
            speaker_df[speaker_df["label"] == "bonafide"]
            .sort_values("utt_id")
            .head(BONAFIDE_PER_SPK)
        )
        rows.extend(bon.to_dict("records"))

        for sys_id in REQUIRED_SYSTEMS:
            spoof = (
                speaker_df[
                    (speaker_df["label"] == "spoof")
                    & (speaker_df["system_id"] == sys_id)
                ]
                .sort_values("utt_id")
                .head(SPOOF_PER_SYSTEM_PER_SPK)
            )
            if len(spoof) != SPOOF_PER_SYSTEM_PER_SPK:
                raise RuntimeError(
                    f"Speaker {speaker_id} missing enough rows for {sys_id}"
                )
            rows.extend(spoof.to_dict("records"))

    selected_df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"Selected utterances: {len(selected_df)}")

    expected_bon = NUM_SPEAKERS * BONAFIDE_PER_SPK
    expected_spoof = NUM_SPEAKERS * len(REQUIRED_SYSTEMS) * SPOOF_PER_SYSTEM_PER_SPK

    bon_count = int((selected_df["label"] == "bonafide").sum())
    spoof_count = int((selected_df["label"] == "spoof").sum())

    assert bon_count == expected_bon, (bon_count, expected_bon)
    assert spoof_count == expected_spoof, (spoof_count, expected_spoof)

    tar_files = sorted(AUDIO_TARS_ROOT.glob("*.tar"))
    if not tar_files:
        raise RuntimeError(f"No tar files found under {AUDIO_TARS_ROOT}")

    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    missing = []

    for idx, row in selected_df.iterrows():
        speaker_id = row["speaker_id"]
        utt_id = row["utt_id"]
        target_path = OUT_AUDIO_DIR / speaker_id / f"{utt_id}.flac"

        if not target_path.exists():
            ok = exact_extract_from_tars(utt_id, target_path, tar_files)
            if not ok:
                missing.append(utt_id)
                print(f"[MISS] {utt_id}")
                continue

        if (idx + 1) % 25 == 0 or idx == len(selected_df) - 1:
            print(f"Prepared {idx + 1}/{len(selected_df)}")

    if missing:
        raise RuntimeError(f"Missing {len(missing)} files, examples: {missing[:10]}")

    final_rows = []
    for _, row in selected_df.iterrows():
        speaker_id = row["speaker_id"]
        utt_id = row["utt_id"]
        audio_path = OUT_AUDIO_DIR / speaker_id / f"{utt_id}.flac"
        if not audio_path.exists():
            continue

        final_rows.append(
            {
                "path": str(audio_path),
                "speaker_id": row["speaker_id"],
                "utt_id": row["utt_id"],
                "label": row["label"],
                "system_id": row["system_id"],
                "gender": row.get("gender", "M"),
                "split": row.get("split", "train"),
                "source_manifest": str(MANIFEST),
            }
        )

    final_df = pd.DataFrame(final_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"CSV: {OUT_CSV}")
    print(f"Rows: {len(final_df)}")
    print("\nLabel counts:")
    print(final_df["label"].value_counts().to_string())
    print("\nSystem counts:")
    print(final_df["system_id"].value_counts().sort_index().to_string())
    print("\nFirst 10 rows:")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
