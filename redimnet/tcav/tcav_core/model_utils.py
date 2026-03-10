"""Model loading and setup utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchaudio

from tcav_core.config import ExperimentConfig
from tcav_core.frame import FrameNormalizer
from tcav_core.modeling import ReDimNetMelLogitsWrapper, SpeakerHead
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
        (head_module, speaker_to_id_mapping, l2_norm_enabled)
    """
    checkpoint = torch.load(head_path, map_location=device, weights_only=False)
    speaker_to_id = checkpoint["speaker_to_id"]
    l2_norm_emb = bool(checkpoint.get("l2_norm_emb", True))

    fc_weight = checkpoint["state_dict"]["fc.weight"]
    in_dim = int(fc_weight.shape[1])
    num_classes = int(fc_weight.shape[0])

    head = SpeakerHead(in_dim=in_dim, num_classes=num_classes).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    return head, speaker_to_id, l2_norm_emb


def setup_models(
    config: ExperimentConfig, tcav_device: torch.device
) -> tuple[Any, ReDimNetMelLogitsWrapper, dict[str, int], FrameNormalizer, int, int]:
    """Load and setup all models and preprocessors.

    Returns:
        (redim_model, wrapped_model, speaker_to_id, frame_normalizer, n_mels, target_frames)
    """
    run_device = str(tcav_device)
    print(f"Initial device: {run_device}")

    # Load backbone
    redim_model = load_redimnet_model(tcav_device)

    # Infer mel dimensions
    with torch.no_grad():
        dummy_wav = torch.zeros(1, 16000, device=tcav_device)
        dummy_mel = redim_model.spec(dummy_wav)
    n_mels = int(dummy_mel.shape[1])

    # Load speaker head
    head_path = abs_path(config.head_path)
    head, speaker_to_id, l2_norm_emb = load_speaker_head(head_path, tcav_device)

    # Wrap model
    wrapped_model = ReDimNetMelLogitsWrapper(
        redim_model,
        head,
        l2_norm_emb=l2_norm_emb,
    ).to(tcav_device)
    wrapped_model.eval()

    # Infer target frames from concepts
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

    # Verify model works
    with torch.no_grad():
        probe = torch.zeros(
            1, 1, n_mels, target_frames, dtype=torch.float32, device=tcav_device
        )
        _ = wrapped_model(probe)

    print(f"N_MELS: {n_mels}")
    print(f"TARGET_FRAMES: {target_frames}")
    print(f"TCAV device: {tcav_device}")
    print("Model forward check: OK")

    return (
        redim_model,
        wrapped_model,
        speaker_to_id,
        frame_normalizer,
        n_mels,
        target_frames,
    )
