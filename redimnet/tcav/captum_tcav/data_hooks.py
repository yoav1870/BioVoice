# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy.typing as npt

import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import override

from captum_tcav import config


@dataclass(frozen=True)
class Config:
    concept_root: Path
    concept_names: list[str]
    random_concept_name: str
    data_dir: Path
    head_path: Path
    target_index: int
    target_mode: str
    model_repo: str
    model_entrypoint: str
    model_name: str
    model_train_type: str
    model_dataset: str
    layers: list[str]
    random_seed: int
    max_clips_per_speaker: int = 32
    max_clips_per_chunk: int = 8


class TorchaudioModule(Protocol):
    def load(self, path: str) -> tuple[torch.Tensor, int]: ...


class SpeakerHead(torch.nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc: torch.nn.Linear = torch.nn.Linear(in_dim, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


class ReDimNetMelLogitsWrapper(torch.nn.Module):
    def __init__(
        self,
        redim_model: torch.nn.Module,
        head: torch.nn.Module,
        l2_norm_emb: bool,
    ) -> None:
        super().__init__()
        self.backbone: torch.nn.Module = cast(
            torch.nn.Module, getattr(redim_model, "backbone")
        )
        self.pool: torch.nn.Module = cast(torch.nn.Module, getattr(redim_model, "pool"))
        self.bn: torch.nn.Module = cast(torch.nn.Module, getattr(redim_model, "bn"))
        self.linear: torch.nn.Module = cast(
            torch.nn.Module, getattr(redim_model, "linear")
        )
        self.head: torch.nn.Module = head
        self.l2_norm_emb: bool = bool(l2_norm_emb)
        self.spec: torch.nn.Module = cast(torch.nn.Module, getattr(redim_model, "spec"))

    @override
    def forward(self, mel4d: torch.Tensor) -> torch.Tensor:
        x = cast(torch.Tensor, self.backbone(mel4d))
        x = cast(torch.Tensor, self.pool(x))
        x = cast(torch.Tensor, self.bn(x))
        emb = cast(torch.Tensor, self.linear(x))
        if self.l2_norm_emb:
            emb = emb / (emb.norm(p=2, dim=1, keepdim=True) + 1e-12)
        return cast(torch.Tensor, self.head(emb))


def module_name_in_model(model: torch.nn.Module, target_module: torch.nn.Module) -> str:
    for name, module in model.named_modules():
        if module is target_module:
            return name
    raise RuntimeError("Could not find target module in model.named_modules().")


def resolve_layers(model: torch.nn.Module, layer_keys: list[str]) -> list[str]:
    layer_map: dict[str, torch.nn.Module] = {}

    backbone_obj = getattr(model, "backbone", None)
    if isinstance(backbone_obj, torch.nn.Module):
        for key in ("stage0", "stage1", "stage2", "stage3", "stage4", "stage5"):
            stage_obj = getattr(backbone_obj, key, None)
            if (
                isinstance(stage_obj, (torch.nn.ModuleList, torch.nn.Sequential))
                and len(stage_obj) > 2
            ):
                stage_module = stage_obj[2]
                layer_map[key] = stage_module
        stem_obj = getattr(backbone_obj, "stem", None)
        if (
            isinstance(stem_obj, (torch.nn.ModuleList, torch.nn.Sequential))
            and len(stem_obj) > 0
        ):
            stem_module = stem_obj[0]
            layer_map["stem"] = stem_module

    resolved: list[str] = []
    for key in layer_keys:
        if "." in key:
            resolved.append(key)
            continue
        if key not in layer_map:
            raise ValueError(
                f"Unknown layer key: {key}. Available keys: {sorted(layer_map.keys())}"
            )
        resolved.append(module_name_in_model(model, layer_map[key]))
    return resolved


def load_config() -> Config:
    return Config(
        concept_root=Path(config.concept_root),
        concept_names=list(config.concept_names),
        random_concept_name=config.random_concept_name,
        data_dir=Path(config.data_dir),
        head_path=Path(config.head_path),
        target_index=int(config.target_index),
        target_mode=str(config.target_mode),
        model_repo=config.model_repo,
        model_entrypoint=config.model_entrypoint,
        model_name=config.model_name,
        model_train_type=config.model_train_type,
        model_dataset=config.model_dataset,
        layers=list(config.layers),
        random_seed=int(config.random_seed),
        max_clips_per_speaker=int(config.max_clips_per_speaker),
        max_clips_per_chunk=int(config.max_clips_per_chunk),
    )


def chunk_paths(paths: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [paths[idx : idx + chunk_size] for idx in range(0, len(paths), chunk_size)]


def concept_dir(config: Config, name: str) -> Path:
    return config.concept_root / name


def discover_speakers(data_dir: Path) -> dict[str, list[Path]]:
    speakers: dict[str, list[Path]] = {}
    for speaker_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        audio_paths = sorted(speaker_dir.rglob("*.wav"))
        audio_paths.extend(sorted(speaker_dir.rglob("*.m4a")))
        audio_paths = sorted(audio_paths)
        if audio_paths:
            speakers[speaker_dir.name] = audio_paths
    return speakers


def concept_npy_paths(config: Config, name: str) -> list[Path]:
    filenames = sorted(concept_dir(config, name).glob("*.npy"))
    if not filenames:
        raise FileNotFoundError(f"No .npy files found for concept: {name}")
    return filenames


def infer_target_frames(config: Config) -> int:
    first_concept_path = concept_npy_paths(config, config.concept_names[0])[0]
    return int(cast(npt.NDArray[np.generic], np.load(first_concept_path)).shape[1])


def normalize_frames(mel_3d: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = int(mel_3d.shape[-1])
    if current_frames == target_frames:
        return mel_3d
    if current_frames > target_frames:
        start = (current_frames - target_frames) // 2
        return mel_3d[..., start : start + target_frames]

    pad = target_frames - current_frames
    left = pad // 2
    right = pad - left
    return F.pad(mel_3d, (left, right), mode="constant", value=0.0)


def ensure_random_concept(config: Config) -> list[Path]:
    random_dir = concept_dir(config, config.random_concept_name)
    existing = sorted(random_dir.glob("*.npy")) if random_dir.is_dir() else []
    reference_paths = concept_npy_paths(config, config.concept_names[0])
    target_count = max(2, len(reference_paths))

    if len(existing) >= target_count:
        return existing

    first_concept_path = reference_paths[0]
    reference_array = cast(npt.NDArray[np.generic], np.load(first_concept_path))
    reference = np.asarray(reference_array, dtype=np.float32)
    rng = np.random.default_rng(config.random_seed)

    random_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(existing), target_count):
        noise = rng.normal(0.0, 1.0, size=reference.shape).astype(np.float32)
        output_path = random_dir / f"gaussian_noise_{idx:03d}.npy"
        np.save(output_path, noise)

    return sorted(random_dir.glob("*.npy"))


def validate_config(config: Config) -> None:
    if not config.concept_names:
        raise ValueError("concept_names must not be empty")
    if config.target_mode != "predicted":
        raise ValueError("target_mode must be 'predicted'")
    if not config.concept_root.is_dir():
        raise FileNotFoundError(f"Missing concept root: {config.concept_root}")
    for name in config.concept_names:
        path = concept_dir(config, name)
        if not path.is_dir():
            raise FileNotFoundError(f"Missing concept folder: {path}")
        _ = concept_npy_paths(config, name)
    if not config.data_dir.is_dir():
        raise FileNotFoundError(f"Missing data dir: {config.data_dir}")
    if not config.head_path.exists():
        raise FileNotFoundError(f"Missing head path: {config.head_path}")


def load_npy_as_tensor(filename: str) -> torch.Tensor:
    array = cast(npt.NDArray[np.generic], np.load(filename))
    return torch.from_numpy(array).float().unsqueeze(0)


def make_iter(name: str, config: Config):
    from captum.concept._utils.data_iterator import (
        CustomIterableDataset,
        dataset_to_dataloader,
    )

    target_frames = infer_target_frames(config)

    def loader(filename: str) -> torch.Tensor:
        return normalize_frames(load_npy_as_tensor(filename), target_frames)

    if name == config.random_concept_name:
        _ = ensure_random_concept(config)
    else:
        _ = concept_npy_paths(config, name)
    dataset = CustomIterableDataset(
        loader,
        os.path.join(str(concept_dir(config, name)), ""),
    )
    return dataset_to_dataloader(dataset)


def load_input(
    config: Config, model: torch.nn.Module, wav_paths: list[Path]
) -> torch.Tensor:
    torchaudio = cast(
        TorchaudioModule,
        cast(object, importlib.import_module("torchaudio")),
    )
    spec = cast(torch.nn.Module, getattr(model, "spec"))
    target_frames = infer_target_frames(config)
    mel_batch: list[torch.Tensor] = []
    for wav_path in wav_paths:
        waveform, _sample_rate = torchaudio.load(str(wav_path))
        waveform = waveform[:1, :].float()
        mel = cast(torch.Tensor, spec(waveform))
        mel_batch.append(normalize_frames(mel, target_frames))
    return torch.stack(mel_batch)


def load_speaker_head(
    head_path: Path, device: torch.device
) -> tuple[SpeakerHead, dict[str, int], bool]:
    checkpoint = cast(
        dict[str, object],
        torch.load(head_path, map_location=device, weights_only=False),
    )
    speaker_to_id = cast(dict[str, int], checkpoint["speaker_to_id"])
    l2_norm_emb = bool(checkpoint.get("l2_norm_emb", True))
    state_dict = cast(dict[str, torch.Tensor], checkpoint["state_dict"])
    fc_weight = state_dict["fc.weight"]
    head = SpeakerHead(int(fc_weight.shape[1]), int(fc_weight.shape[0])).to(device)
    _ = head.load_state_dict(state_dict)
    _ = head.eval()
    return head, speaker_to_id, l2_norm_emb


def predict_target_index(model: torch.nn.Module, inputs: torch.Tensor) -> int:
    with torch.no_grad():
        logits = cast(torch.Tensor, model(inputs))
    predicted = torch.argmax(logits, dim=1)
    return int(predicted[0].item())


def load_model(config: Config) -> torch.nn.Module:
    raw_model = cast(
        torch.nn.Module,
        torch.hub.load(
            config.model_repo,
            config.model_entrypoint,
            model_name=config.model_name,
            train_type=config.model_train_type,
            dataset=config.model_dataset,
        ),
    )
    try:
        device = next(raw_model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    head, _speaker_to_id, l2_norm_emb = load_speaker_head(config.head_path, device)
    return ReDimNetMelLogitsWrapper(raw_model.eval(), head, l2_norm_emb).eval()
