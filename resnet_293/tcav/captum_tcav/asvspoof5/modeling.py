# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import importlib.util
import pickle
import sys
import types
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from .config import Config
from .data import model_dir


def _load_wespeaker_resnet293_class(wespeaker_models_dir: Path) -> type:
    """Dynamically load the WeSpeaker ResNet293 constructor from local files."""
    # Register stub packages so imports inside the WeSpeaker files resolve.
    if "wespeaker" not in sys.modules:
        sys.modules["wespeaker"] = types.ModuleType("wespeaker")
    if "wespeaker.models" not in sys.modules:
        sys.modules["wespeaker.models"] = types.ModuleType("wespeaker.models")

    pool_spec = importlib.util.spec_from_file_location(
        "wespeaker.models.pooling_layers",
        str(wespeaker_models_dir / "pooling_layers.py"),
    )
    if pool_spec is None or pool_spec.loader is None:
        raise ImportError("Cannot find wespeaker pooling_layers.py")
    pool_mod = importlib.util.module_from_spec(pool_spec)
    sys.modules["wespeaker.models.pooling_layers"] = pool_mod
    pool_spec.loader.exec_module(pool_mod)

    resnet_spec = importlib.util.spec_from_file_location(
        "wespeaker.models.resnet",
        str(wespeaker_models_dir / "resnet.py"),
    )
    if resnet_spec is None or resnet_spec.loader is None:
        raise ImportError("Cannot find wespeaker resnet.py")
    resnet_mod = importlib.util.module_from_spec(resnet_spec)
    sys.modules["wespeaker.models.resnet"] = resnet_mod
    resnet_spec.loader.exec_module(resnet_mod)

    return resnet_mod.ResNet293


class ResNet293WithSpoofLogit(torch.nn.Module):
    """Wraps a WeSpeaker ResNet293 with a logistic-regression spoof head.

    The forward pass computes the speaker embedding and then applies the
    scaler + logistic-regression classifier (stored as frozen buffer tensors)
    to produce a single spoof logit per sample.
    """

    def __init__(
        self,
        resnet_model: torch.nn.Module,
        scaler: StandardScaler,
        classifier: LogisticRegression,
    ) -> None:
        super().__init__()
        # Expose the convolution stack so TCAV can hook into named modules.
        self.conv1: torch.nn.Module = cast(
            torch.nn.Module, getattr(resnet_model, "conv1")
        )
        self.bn1: torch.nn.Module = cast(torch.nn.Module, getattr(resnet_model, "bn1"))
        self.layer1: torch.nn.Module = cast(
            torch.nn.Module, getattr(resnet_model, "layer1")
        )
        self.layer2: torch.nn.Module = cast(
            torch.nn.Module, getattr(resnet_model, "layer2")
        )
        self.layer3: torch.nn.Module = cast(
            torch.nn.Module, getattr(resnet_model, "layer3")
        )
        self.layer4: torch.nn.Module = cast(
            torch.nn.Module, getattr(resnet_model, "layer4")
        )
        self.pool: torch.nn.Module = cast(
            torch.nn.Module, getattr(resnet_model, "pool")
        )
        self.seg_1: torch.nn.Module = cast(
            torch.nn.Module, getattr(resnet_model, "seg_1")
        )

        # Store the two_emb_layer flag and optional second embedding layer.
        self._two_emb_layer: bool = bool(getattr(resnet_model, "two_emb_layer", False))
        if self._two_emb_layer:
            self.seg_bn_1: torch.nn.Module = cast(
                torch.nn.Module, getattr(resnet_model, "seg_bn_1")
            )
            self.seg_2: torch.nn.Module = cast(
                torch.nn.Module, getattr(resnet_model, "seg_2")
            )

        # Bake logistic-regression weights into buffer tensors.
        coef = torch.tensor(classifier.coef_[0], dtype=torch.float32)
        intercept = torch.tensor(float(classifier.intercept_[0]), dtype=torch.float32)
        scale = torch.tensor(scaler.scale_, dtype=torch.float32)
        mean = torch.tensor(scaler.mean_, dtype=torch.float32)

        if coef.shape[0] != scale.shape[0] or coef.shape[0] != mean.shape[0]:
            raise ValueError("Scaler and logistic regression dimensions do not match")

        self.register_buffer("lr_coef", coef)
        self.register_buffer("lr_intercept", intercept)
        self.register_buffer("scaler_scale", scale)
        self.register_buffer("scaler_mean", mean)

    def _get_frame_level_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Reproduce ResNet._get_frame_level_feat: (B,T,F) -> 4-D feature map."""
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze(1)  # (B,1,F,T)
        out = F.relu(self.bn1(self.conv1(x)))
        out = cast(torch.Tensor, self.layer1(out))
        out = cast(torch.Tensor, self.layer2(out))
        out = cast(torch.Tensor, self.layer3(out))
        out = cast(torch.Tensor, self.layer4(out))
        return out

    @override
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, F) -- e.g. (B, 200, 80) fbank features."""
        out = self._get_frame_level_feat(feats)
        stats = cast(torch.Tensor, self.pool(out))
        emb = cast(torch.Tensor, self.seg_1(stats))

        if self._two_emb_layer:
            emb = F.relu(emb)
            emb = cast(torch.Tensor, self.seg_bn_1(emb))
            emb = cast(torch.Tensor, self.seg_2(emb))

        # Logistic-regression spoof head.
        scaled = (emb - self.scaler_mean.unsqueeze(0)) / self.scaler_scale.unsqueeze(0)
        logits = (
            torch.sum(scaled * self.lr_coef.unsqueeze(0), dim=1) + self.lr_intercept
        )
        return logits.unsqueeze(1)


def load_logistic_artifacts(
    task_model_dir: Path,
) -> tuple[StandardScaler, LogisticRegression]:
    with (task_model_dir / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (task_model_dir / "logistic_regression.pkl").open("rb") as handle:
        classifier = pickle.load(handle)
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Expected scaler.pkl to contain sklearn StandardScaler")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError(
            "Expected logistic_regression.pkl to contain sklearn LogisticRegression"
        )
    return scaler, classifier


def _load_raw_resnet293(config: Config, device: torch.device) -> torch.nn.Module:
    """Instantiate ResNet293 from WeSpeaker files and load pre-trained weights."""
    import yaml

    ResNet293Cls = _load_wespeaker_resnet293_class(config.wespeaker_models_dir)

    with (config.wespeaker_model_dir / "config.yaml").open("r", encoding="utf-8") as f:
        wespeaker_config = yaml.safe_load(f)

    model = ResNet293Cls(**wespeaker_config["model_args"]).to(device)

    ckpt_path = config.wespeaker_model_dir / "avg_model.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
    if isinstance(state, dict):
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    _ = model.load_state_dict(state, strict=False)
    return model.eval()


def load_model(config: Config, system_id: str) -> torch.nn.Module:
    device = torch.device("cpu")
    raw_model = _load_raw_resnet293(config, device)
    logistic_dir = (
        config.global_model_dir
        if config.model_loading_mode == "global"
        else model_dir(config, system_id)
    )
    scaler, classifier = load_logistic_artifacts(logistic_dir)
    return ResNet293WithSpoofLogit(raw_model, scaler, classifier).eval()
