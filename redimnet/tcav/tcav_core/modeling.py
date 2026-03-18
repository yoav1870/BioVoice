from __future__ import annotations

import torch
import torch.nn as nn


class SpeakerHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SpoofLogRegHead(nn.Module):
    def __init__(
        self,
        coef: torch.Tensor,
        intercept: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("coef", coef.reshape(1, -1).float())
        self.register_buffer("intercept", intercept.reshape(1).float())
        self.register_buffer("mean", mean.reshape(1, -1).float())
        self.register_buffer("scale", scale.reshape(1, -1).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_std = (x - self.mean) / torch.clamp(self.scale, min=1e-12)
        spoof_logit = x_std @ self.coef.transpose(0, 1) + self.intercept
        spoof_logit = spoof_logit.squeeze(1)
        zero_logit = torch.zeros_like(spoof_logit)
        return torch.stack([zero_logit, spoof_logit], dim=1)


class ReDimNetMelLogitsWrapper(nn.Module):
    def __init__(self, redim_model, head: nn.Module, l2_norm_emb: bool) -> None:
        super().__init__()
        self.backbone = redim_model.backbone
        self.pool = redim_model.pool
        self.bn = redim_model.bn
        self.linear = redim_model.linear
        self.head = head
        self.l2_norm_emb = bool(l2_norm_emb)

    def forward(self, mel4d: torch.Tensor) -> torch.Tensor:
        x = self.backbone(mel4d)
        x = self.pool(x)
        x = self.bn(x)
        emb = self.linear(x)
        if self.l2_norm_emb:
            emb = emb / (emb.norm(p=2, dim=1, keepdim=True) + 1e-12)
        return self.head(emb)


def module_name_in_model(model: nn.Module, target_module: nn.Module) -> str:
    for name, module in model.named_modules():
        if module is target_module:
            return name
    raise RuntimeError("Could not find target module in wrapped_model.named_modules().")


def build_target_layers(
    wrapped_model: ReDimNetMelLogitsWrapper,
) -> dict[str, nn.Module]:
    return {
        "stem": wrapped_model.backbone.stem[0],
        "stage0": wrapped_model.backbone.stage0[2],
        "stage1": wrapped_model.backbone.stage1[2],
        "stage2": wrapped_model.backbone.stage2[2],
        "stage3": wrapped_model.backbone.stage3[2],
        "stage4": wrapped_model.backbone.stage4[2],
        "stage5": wrapped_model.backbone.stage5[2],
    }
