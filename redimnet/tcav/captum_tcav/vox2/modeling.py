# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import torch
from typing_extensions import override

from .config import Config


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


class SpeakerHeadCheckpoint(Protocol):
    speaker_to_id: dict[str, int]
    l2_norm_emb: bool
    state_dict: dict[str, torch.Tensor]


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
