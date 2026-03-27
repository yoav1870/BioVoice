# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import io
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from captum_tcav.concepts import infer_target_frames, normalize_frames

from .config import Config


@dataclass(frozen=True)
class SpoofSubset:
    selected_rows: pd.DataFrame
    selected_speakers: list[str]


def manifest_path(config: Config) -> Path:
    return config.plan_base / config.source_partition / "selected_utterances_plan.csv"


def model_dir(config: Config) -> Path:
    return config.trained_models_base / config.source_partition / config.system_id


def validate_config(config: Config) -> None:
    if not config.concept_names:
        raise ValueError("concept_names must not be empty")
    if not config.concept_root.is_dir():
        raise FileNotFoundError(f"Missing concept root: {config.concept_root}")
    for name in config.concept_names:
        path = config.concept_root / name
        if not path.is_dir():
            raise FileNotFoundError(f"Missing concept folder: {path}")
    if config.subset_num_speakers <= 0:
        raise ValueError("subset_num_speakers must be positive")
    if config.subset_utts_per_speaker <= 0:
        raise ValueError("subset_utts_per_speaker must be positive")
    if config.subset_min_utts_per_speaker < config.subset_utts_per_speaker:
        raise ValueError(
            "subset_min_utts_per_speaker must be >= subset_utts_per_speaker"
        )
    m_path = manifest_path(config)
    if not m_path.exists():
        raise FileNotFoundError(f"Missing manifest: {m_path}")
    m_dir = model_dir(config)
    if not (m_dir / "scaler.pkl").exists():
        raise FileNotFoundError(f"Missing scaler artifact: {m_dir / 'scaler.pkl'}")
    if not (m_dir / "logistic_regression.pkl").exists():
        raise FileNotFoundError(
            f"Missing logistic artifact: {m_dir / 'logistic_regression.pkl'}"
        )


def load_manifest(config: Config) -> pd.DataFrame:
    df = pd.read_csv(manifest_path(config))
    required_cols = {
        "split",
        "speaker_id",
        "utt_id",
        "gender",
        "label",
        "system_id",
        "sample_class",
        "target_class",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return df


def select_spoof_subset(config: Config, manifest: pd.DataFrame) -> SpoofSubset:
    spoof_df = manifest[
        manifest["split"].eq(config.split_name)
        & manifest["target_class"].eq(config.system_id)
    ].copy()
    if spoof_df.empty:
        raise RuntimeError(
            f"No rows found for system={config.system_id} split={config.split_name}"
        )

    speaker_counts = (
        spoof_df.groupby("speaker_id")
        .agg(n_utterances=("utt_id", "size"))
        .reset_index()
    )
    eligible = speaker_counts[
        speaker_counts["n_utterances"] >= config.subset_min_utts_per_speaker
    ].copy()
    if len(eligible) < config.subset_num_speakers:
        raise RuntimeError(
            "Not enough eligible speakers for subset selection: "
            f"need {config.subset_num_speakers}, found {len(eligible)}"
        )

    rng = np.random.default_rng(config.subset_seed)
    chosen_speakers = sorted(
        rng.choice(
            eligible["speaker_id"].to_numpy(),
            size=config.subset_num_speakers,
            replace=False,
        ).tolist()
    )

    selected_parts: list[pd.DataFrame] = []
    for offset, speaker_id in enumerate(chosen_speakers):
        speaker_rows = spoof_df[spoof_df["speaker_id"].eq(speaker_id)].copy()
        speaker_rows = speaker_rows.sample(
            n=config.subset_utts_per_speaker,
            random_state=config.subset_seed + offset,
            replace=False,
        ).sort_values("utt_id")
        selected_parts.append(speaker_rows)

    selected = pd.concat(selected_parts, axis=0, ignore_index=True)
    return SpoofSubset(selected_rows=selected, selected_speakers=chosen_speakers)


def _resolve_tar_dir_and_prefix(config: Config) -> tuple[Path, str]:
    tar_root = config.plan_base.parents[1]
    if config.source_partition == "train":
        return (
            tar_root / "ASVspoof5_audio_train_tars",
            "flac_T_*.tar",
        )
    if config.source_partition == "dev":
        return (
            tar_root / "ASVspoof5_audio_dev_tars",
            "flac_D_*.tar",
        )
    raise ValueError(f"Unsupported source_partition: {config.source_partition}")


def _build_tar_member_index(tar_dir: Path, tar_prefix: str) -> dict[str, tuple[Path, str]]:
    tar_paths = sorted(tar_dir.glob(tar_prefix))
    if not tar_paths:
        raise FileNotFoundError(f"No tar files found in {tar_dir} matching {tar_prefix}")

    index: dict[str, tuple[Path, str]] = {}
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as handle:
            for member in handle:
                if not member.isfile():
                    continue
                utt_id = Path(member.name).stem
                if utt_id not in index:
                    index[utt_id] = (tar_path, member.name)
    return index


def _load_waveform_from_tar(tar_path: Path, member_name: str) -> tuple[torch.Tensor, int]:
    with tarfile.open(tar_path, "r") as handle:
        member = handle.getmember(member_name)
        extracted = handle.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not extract {member_name} from {tar_path}")
        raw = extracted.read()
    try:
        return torchaudio.load(io.BytesIO(raw))
    except Exception:
        suffix = Path(member_name).suffix or ".flac"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()
            return torchaudio.load(tmp.name)


def load_input(
    config: Config,
    model: torch.nn.Module,
    utt_ids: list[str],
    tar_index: dict[str, tuple[Path, str]],
) -> torch.Tensor:
    spec = getattr(model, "spec", None)
    if not isinstance(spec, torch.nn.Module):
        raise TypeError("Expected model.spec to be a torch.nn.Module")
    target_frames = infer_target_frames(config.concept_root, config.concept_names)
    mel_batch: list[torch.Tensor] = []
    for utt_id in utt_ids:
        if utt_id not in tar_index:
            raise KeyError(f"utt_id not found in tar index: {utt_id}")
        tar_path, member_name = tar_index[utt_id]
        waveform, sample_rate = _load_waveform_from_tar(tar_path, member_name)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform[:1, :].float()
        mel = spec(waveform)
        if not isinstance(mel, torch.Tensor):
            raise TypeError("Expected spectrogram output to be a torch.Tensor")
        mel_batch.append(normalize_frames(mel, target_frames))
    return torch.stack(mel_batch)


def build_tar_index(config: Config) -> dict[str, tuple[Path, str]]:
    tar_dir, tar_prefix = _resolve_tar_dir_and_prefix(config)
    return _build_tar_member_index(tar_dir, tar_prefix)
