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
    """Resolve human-friendly layer keys to fully-qualified module names for ResNet293.

    Supported keys: "layer1", "layer2", "layer3", "layer4", "conv1", "bn1", "pool", "seg_1".
    Dotted keys (e.g. "layer4.2") are passed through unchanged.
    """
    layer_map: dict[str, torch.nn.Module] = {}

    for key in ("layer1", "layer2", "layer3", "layer4", "conv1", "bn1", "pool", "seg_1"):
        obj = getattr(model, key, None)
        if isinstance(obj, torch.nn.Module):
            layer_map[key] = obj

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
