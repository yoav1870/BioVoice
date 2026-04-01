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


def partition_for_system(system_id: str) -> str:
    numeric = int(system_id[1:])
    if 1 <= numeric <= 8:
        return "train"
    if 9 <= numeric <= 16:
        return "dev"
    raise ValueError(f"Unsupported system_id: {system_id}")


def manifest_path(config: Config, system_id: str) -> Path:
    return config.plan_base / partition_for_system(system_id) / "selected_utterances_plan.csv"


def model_dir(config: Config, system_id: str) -> Path:
    return config.trained_models_base / partition_for_system(system_id) / system_id


def validate_config(config: Config) -> None:
    if not config.concept_names:
        raise ValueError("concept_names must not be empty")
    if not config.system_ids:
        raise ValueError("system_ids must not be empty")
    if config.example_class not in {"spoof", "bonafide"}:
        raise ValueError("example_class must be 'spoof' or 'bonafide'")
    if config.target_class_mode not in {"single", "both"}:
        raise ValueError("target_class_mode must be 'single' or 'both'")
    if config.model_loading_mode not in {"per_system", "global"}:
        raise ValueError("model_loading_mode must be 'per_system' or 'global'")
    if config.output_mode not in {"mean", "row"}:
        raise ValueError("output_mode must be 'mean' or 'row'")
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
    for system_id in config.system_ids:
        _ = partition_for_system(system_id)
        m_path = manifest_path(config, system_id)
        if not m_path.exists():
            raise FileNotFoundError(f"Missing manifest for {system_id}: {m_path}")
        m_dir = model_dir(config, system_id)
        if not (m_dir / "scaler.pkl").exists():
            raise FileNotFoundError(
                f"Missing scaler artifact for {system_id}: {m_dir / 'scaler.pkl'}"
            )
        if not (m_dir / "logistic_regression.pkl").exists():
            raise FileNotFoundError(
                f"Missing logistic artifact for {system_id}: {m_dir / 'logistic_regression.pkl'}"
            )
    if config.model_loading_mode == "global":
        if not (config.global_model_dir / "scaler.pkl").exists():
            raise FileNotFoundError(
                f"Missing global scaler artifact: {config.global_model_dir / 'scaler.pkl'}"
            )
        if not (config.global_model_dir / "logistic_regression.pkl").exists():
            raise FileNotFoundError(
                "Missing global logistic artifact: "
                f"{config.global_model_dir / 'logistic_regression.pkl'}"
            )


def fixed_speakers_for_partition(config: Config, partition: str) -> list[str]:
    if partition == "train":
        return list(config.fixed_train_speakers or [])
    if partition == "dev":
        return list(config.fixed_dev_speakers or [])
    raise ValueError(f"Unsupported partition: {partition}")


def excluded_speakers_for_partition(config: Config, partition: str) -> list[str]:
    if partition == "train":
        return list(config.excluded_train_speakers or [])
    if partition == "dev":
        return list(config.excluded_dev_speakers or [])
    raise ValueError(f"Unsupported partition: {partition}")


def load_manifest(config: Config, system_id: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path(config, system_id))
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


def select_subset_for_class(
    config: Config,
    system_id: str,
    manifest: pd.DataFrame,
    example_class: str,
    forced_speakers: list[str] | None = None,
) -> SpoofSubset:
    partition = partition_for_system(system_id)
    target_class = system_id if example_class == "spoof" else "bonafide"
    subset_df = manifest[
        manifest["split"].eq(config.split_name)
        & manifest["target_class"].eq(target_class)
    ].copy()
    if subset_df.empty:
        raise RuntimeError(
            f"No rows found for system={system_id} class={target_class} split={config.split_name}"
        )

    fixed_speakers = (
        list(forced_speakers) if forced_speakers is not None else fixed_speakers_for_partition(config, partition)
    )
    excluded_speakers = excluded_speakers_for_partition(config, partition)

    if excluded_speakers:
        subset_df = subset_df[~subset_df["speaker_id"].isin(excluded_speakers)].copy()
        if subset_df.empty:
            raise RuntimeError(
                f"No rows left after applying excluded speaker list for partition={partition}"
            )

    if fixed_speakers:
        overlap = sorted(set(fixed_speakers).intersection(excluded_speakers))
        if overlap:
            raise RuntimeError(
                f"Fixed and excluded speaker lists overlap for partition={partition}: {overlap}"
            )
        subset_df = subset_df[subset_df["speaker_id"].isin(fixed_speakers)].copy()
        if subset_df.empty:
            raise RuntimeError(
                f"No rows left after applying fixed speaker list for partition={partition}"
            )

    speaker_counts = (
        subset_df.groupby("speaker_id")
        .agg(n_utterances=("utt_id", "size"))
        .reset_index()
    )
    eligible = speaker_counts[
        speaker_counts["n_utterances"] >= config.subset_min_utts_per_speaker
    ].copy()

    if fixed_speakers:
        missing_fixed = sorted(set(fixed_speakers) - set(eligible["speaker_id"].tolist()))
        if missing_fixed:
            raise RuntimeError(
                f"Fixed speakers do not satisfy min utterance threshold for partition={partition}: {missing_fixed}"
            )
        chosen_speakers = list(fixed_speakers)
        if len(chosen_speakers) != config.subset_num_speakers:
            raise RuntimeError(
                "subset_num_speakers must match the fixed speaker list length "
                f"for partition={partition}: expected {len(chosen_speakers)}, got {config.subset_num_speakers}"
            )
    else:
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
        speaker_rows = subset_df[subset_df["speaker_id"].eq(speaker_id)].copy()
        speaker_rows = speaker_rows.sample(
            n=config.subset_utts_per_speaker,
            random_state=config.subset_seed + offset,
            replace=False,
        ).sort_values("utt_id")
        selected_parts.append(speaker_rows)

    selected = pd.concat(selected_parts, axis=0, ignore_index=True)
    return SpoofSubset(selected_rows=selected, selected_speakers=chosen_speakers)


def shared_speakers_for_both_classes(
    config: Config,
    system_id: str,
    manifest: pd.DataFrame,
) -> list[str]:
    partition = partition_for_system(system_id)
    excluded_speakers = excluded_speakers_for_partition(config, partition)
    fixed_speakers = fixed_speakers_for_partition(config, partition)

    def eligible_speakers(example_class: str) -> set[str]:
        target_class = system_id if example_class == "spoof" else "bonafide"
        subset_df = manifest[
            manifest["split"].eq(config.split_name)
            & manifest["target_class"].eq(target_class)
        ].copy()
        if excluded_speakers:
            subset_df = subset_df[~subset_df["speaker_id"].isin(excluded_speakers)].copy()
        speaker_counts = (
            subset_df.groupby("speaker_id")
            .agg(n_utterances=("utt_id", "size"))
            .reset_index()
        )
        eligible = speaker_counts[
            speaker_counts["n_utterances"] >= config.subset_min_utts_per_speaker
        ]["speaker_id"].astype(str)
        return set(eligible.tolist())

    shared = sorted(eligible_speakers("spoof").intersection(eligible_speakers("bonafide")))
    if fixed_speakers:
        shared = [speaker for speaker in fixed_speakers if speaker in set(shared)]
    if len(shared) < config.subset_num_speakers:
        raise RuntimeError(
            "Not enough eligible shared speakers between spoof and bonafide for "
            f"{system_id}: need {config.subset_num_speakers}, found {len(shared)}"
        )
    if fixed_speakers:
        if len(shared) != config.subset_num_speakers:
            raise RuntimeError(
                "subset_num_speakers must match the fixed shared speaker list length "
                f"for partition={partition}: expected {len(shared)}, got {config.subset_num_speakers}"
            )
        return shared

    rng = np.random.default_rng(config.subset_seed)
    chosen = sorted(
        rng.choice(
            np.array(shared, dtype=object),
            size=config.subset_num_speakers,
            replace=False,
        ).tolist()
    )
    return chosen


def select_subset(config: Config, system_id: str, manifest: pd.DataFrame) -> SpoofSubset:
    return select_subset_for_class(config, system_id, manifest, config.example_class)


def _resolve_tar_dir_and_prefix(config: Config, system_id: str) -> tuple[Path, str]:
    tar_root = config.plan_base.parents[1]
    partition = partition_for_system(system_id)
    if partition == "train":
        return (
            tar_root / "ASVspoof5_audio_train_tars",
            "flac_T_*.tar",
        )
    if partition == "dev":
        return (
            tar_root / "ASVspoof5_audio_dev_tars",
            "flac_D_*.tar",
        )
    raise ValueError(f"Unsupported source_partition: {partition}")


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


def build_tar_index(config: Config, system_id: str) -> dict[str, tuple[Path, str]]:
    tar_dir, tar_prefix = _resolve_tar_dir_and_prefix(config, system_id)
    return _build_tar_member_index(tar_dir, tar_prefix)
