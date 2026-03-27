from __future__ import annotations

from typing import cast

import torch


def chunk_paths(paths: list, chunk_size: int) -> list[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [paths[idx : idx + chunk_size] for idx in range(0, len(paths), chunk_size)]


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


def predict_argmax_target(model: torch.nn.Module, inputs: torch.Tensor) -> int:
    with torch.no_grad():
        logits = cast(torch.Tensor, model(inputs))
    predicted = torch.argmax(logits, dim=1)
    return int(predicted[0].item())
