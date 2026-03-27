# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, cast

import torch

from captum_tcav.concepts import infer_target_frames, normalize_frames

from .config import Config


class TorchaudioModule(Protocol):
    def load(self, path: str) -> tuple[torch.Tensor, int]: ...


def discover_speakers(data_dir: Path) -> dict[str, list[Path]]:
    speakers: dict[str, list[Path]] = {}
    for speaker_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        audio_paths = sorted(speaker_dir.rglob("*.wav"))
        audio_paths.extend(sorted(speaker_dir.rglob("*.m4a")))
        audio_paths = sorted(audio_paths)
        if audio_paths:
            speakers[speaker_dir.name] = audio_paths
    return speakers


def validate_config(config: Config) -> None:
    if not config.concept_names:
        raise ValueError("concept_names must not be empty")
    if config.target_mode != "predicted":
        raise ValueError("target_mode must be 'predicted'")
    if not config.concept_root.is_dir():
        raise FileNotFoundError(f"Missing concept root: {config.concept_root}")
    for name in config.concept_names:
        path = config.concept_root / name
        if not path.is_dir():
            raise FileNotFoundError(f"Missing concept folder: {path}")
    if not config.data_dir.is_dir():
        raise FileNotFoundError(f"Missing data dir: {config.data_dir}")
    if not config.head_path.exists():
        raise FileNotFoundError(f"Missing head path: {config.head_path}")


def load_input(
    config: Config, model: torch.nn.Module, wav_paths: list[Path]
) -> torch.Tensor:
    torchaudio = cast(
        TorchaudioModule,
        cast(object, importlib.import_module("torchaudio")),
    )
    spec = cast(torch.nn.Module, getattr(model, "spec"))
    target_frames = infer_target_frames(config.concept_root, config.concept_names)
    mel_batch: list[torch.Tensor] = []
    for wav_path in wav_paths:
        waveform, _sample_rate = torchaudio.load(str(wav_path))
        waveform = waveform[:1, :].float()
        mel = cast(torch.Tensor, spec(waveform))
        mel_batch.append(normalize_frames(mel, target_frames))
    return torch.stack(mel_batch)
