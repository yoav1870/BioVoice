"""Model loading and setup utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchaudio

from tcav_core.config import ExperimentConfig
from tcav_core.frame import FrameNormalizer
from tcav_core.modeling import ReDimNetMelLogitsWrapper, SpeakerHead, SpoofLogRegHead
from tcav_core.utils import abs_path


def load_redimnet_model(
    device: torch.device, model_name: str = "b5", train_type: str = "ptn"
) -> Any:
    """Load ReDimNet backbone from torch hub."""
    return (
        torch.hub.load(
            "IDRnD/ReDimNet",
            "ReDimNet",
            model_name=model_name,
            train_type=train_type,
            dataset="vox2",
        )
        .to(device)
        .eval()
    )


def load_speaker_head(
    head_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, int], bool]:
    """Load speaker classification head from checkpoint.

    Returns:
        (head_module, label_to_id_mapping, l2_norm_enabled)
    """
    checkpoint = torch.load(head_path, map_location=device, weights_only=False)
    label_to_id = checkpoint["speaker_to_id"]
    l2_norm_emb = bool(checkpoint.get("l2_norm_emb", True))

    fc_weight = checkpoint["state_dict"]["fc.weight"]
    in_dim = int(fc_weight.shape[1])
    num_classes = int(fc_weight.shape[0])

    head = SpeakerHead(in_dim=in_dim, num_classes=num_classes).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    return head, label_to_id, l2_norm_emb


def _normalize_spoof_label_to_id(classes: list[Any]) -> dict[str, int]:
    if len(classes) != 2:
        raise RuntimeError(f"Spoof logreg must be binary. Got classes={classes}")

    raw = [str(x) for x in classes]
    raw_set = set(raw)
    if raw_set == {"0", "1"}:
        return {
            "bonafide": 0,
            "spoof": 1,
            "0": 0,
            "1": 1,
        }
    if raw_set == {"bonafide", "spoof"}:
        return {
            "bonafide": raw.index("bonafide"),
            "spoof": raw.index("spoof"),
            "0": raw.index("bonafide"),
            "1": raw.index("spoof"),
        }
    raise RuntimeError(
        "Spoof logreg classes must be either [0,1] or [bonafide,spoof]. "
        f"Got classes={classes}"
    )


def load_spoof_logreg_head(
    logreg_path: Path,
    scaler_path: Path,
    device: torch.device,
    l2_norm_emb: bool,
) -> tuple[nn.Module, dict[str, int], bool]:
    """Load a sklearn logistic-regression spoof detector as a torch head."""
    with open(logreg_path, "rb") as f:
        logreg = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    if not hasattr(logreg, "coef_") or not hasattr(logreg, "intercept_"):
        raise RuntimeError(f"Unsupported spoof logreg artifact: {logreg_path}")
    if not hasattr(logreg, "classes_"):
        raise RuntimeError(f"Spoof logreg missing classes_: {logreg_path}")
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise RuntimeError(f"Unsupported scaler artifact: {scaler_path}")

    label_to_id = _normalize_spoof_label_to_id(logreg.classes_.tolist())

    head = SpoofLogRegHead(
        coef=torch.as_tensor(logreg.coef_[0], device=device),
        intercept=torch.as_tensor(logreg.intercept_[0], device=device),
        mean=torch.as_tensor(scaler.mean_, device=device),
        scale=torch.as_tensor(scaler.scale_, device=device),
    ).to(device)
    head.eval()

    return head, label_to_id, bool(l2_norm_emb)


def load_head_for_config(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, int], bool]:
    head_type = str(config.head_type).strip().lower()
    if head_type == "speaker":
        head_path = abs_path(config.head_path)
        return load_speaker_head(head_path, device)
    if head_type == "spoof_logreg":
        if config.spoof_logreg_path is None or config.spoof_scaler_path is None:
            raise RuntimeError(
                "spoof_logreg head_type requires spoof_logreg_path and spoof_scaler_path."
            )
        logreg_path = abs_path(config.spoof_logreg_path)
        scaler_path = abs_path(config.spoof_scaler_path)
        return load_spoof_logreg_head(
            logreg_path=logreg_path,
            scaler_path=scaler_path,
            device=device,
            l2_norm_emb=config.spoof_l2_norm_emb,
        )
    raise ValueError(
        f"Unsupported head_type={config.head_type!r}. Expected 'speaker' or 'spoof_logreg'."
    )


def setup_models(
    config: ExperimentConfig, tcav_device: torch.device
) -> tuple[Any, ReDimNetMelLogitsWrapper, dict[str, int], FrameNormalizer, int, int]:
    """Load and setup all models and preprocessors.

    Returns:
        (redim_model, wrapped_model, label_to_id, frame_normalizer, n_mels, target_frames)
    """
    run_device = str(tcav_device)
    print(f"Initial device: {run_device}")

    redim_model = load_redimnet_model(tcav_device)

    with torch.no_grad():
        dummy_wav = torch.zeros(1, 16000, device=tcav_device)
        dummy_mel = redim_model.spec(dummy_wav)
    n_mels = int(dummy_mel.shape[1])

    head, label_to_id, l2_norm_emb = load_head_for_config(config, tcav_device)

    wrapped_model = ReDimNetMelLogitsWrapper(
        redim_model,
        head,
        l2_norm_emb=l2_norm_emb,
    ).to(tcav_device)
    wrapped_model.eval()

    concept_root = abs_path(config.concept_root)
    concept_dirs = sorted([d for d in concept_root.iterdir() if d.is_dir()])
    from tcav_core.datasets import infer_target_frames_from_concepts

    if config.target_frames_override is None:
        target_frames = infer_target_frames_from_concepts(concept_dirs)
    else:
        target_frames = int(config.target_frames_override)

    frame_normalizer = FrameNormalizer(
        target_frames=target_frames,
        crop_mode=config.frame_crop_mode,
        pad_mode=config.frame_pad_mode,
    )

    with torch.no_grad():
        probe = torch.zeros(
            1, 1, n_mels, target_frames, dtype=torch.float32, device=tcav_device
        )
        _ = wrapped_model(probe)

    print(f"HEAD_TYPE: {config.head_type}")
    print(f"N_MELS: {n_mels}")
    print(f"TARGET_FRAMES: {target_frames}")
    print(f"TCAV device: {tcav_device}")
    print("Model forward check: OK")

    return (
        redim_model,
        wrapped_model,
        label_to_id,
        frame_normalizer,
        n_mels,
        target_frames,
    )
