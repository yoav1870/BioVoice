# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import pickle
from pathlib import Path
from typing import cast

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from .config import Config
from .data import model_dir


class ReDimNetWithSpoofLogit(torch.nn.Module):
    def __init__(
        self,
        redim_model: torch.nn.Module,
        scaler: StandardScaler,
        classifier: LogisticRegression,
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
        self.spec: torch.nn.Module = cast(torch.nn.Module, getattr(redim_model, "spec"))

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

    @override
    def forward(self, mel4d: torch.Tensor) -> torch.Tensor:
        x = cast(torch.Tensor, self.backbone(mel4d))
        x = cast(torch.Tensor, self.pool(x))
        x = cast(torch.Tensor, self.bn(x))
        emb = cast(torch.Tensor, self.linear(x))
        scaled = (emb - self.scaler_mean.unsqueeze(0)) / self.scaler_scale.unsqueeze(0)
        logits = torch.sum(scaled * self.lr_coef.unsqueeze(0), dim=1) + self.lr_intercept
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


def load_model(config: Config, system_id: str) -> torch.nn.Module:
    redim_model = cast(
        torch.nn.Module,
        torch.hub.load(
            config.model_repo,
            config.model_entrypoint,
            model_name=config.model_name,
            train_type=config.model_train_type,
            dataset=config.model_dataset,
        ),
    ).eval()
    logistic_dir = (
        config.global_model_dir
        if config.model_loading_mode == "global"
        else model_dir(config, system_id)
    )
    scaler, classifier = load_logistic_artifacts(logistic_dir)
    return ReDimNetWithSpoofLogit(redim_model, scaler, classifier).eval()
