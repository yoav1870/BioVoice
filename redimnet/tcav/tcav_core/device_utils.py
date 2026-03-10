"""Device utilities for TCAV runner."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


class DeviceDataLoader:
    """Wrapper that moves batches to specified device during iteration."""

    def __init__(self, data_loader: DataLoader, device: torch.device) -> None:
        self.data_loader = data_loader
        self.device = device

    def __iter__(self):
        for batch in self.data_loader:
            yield _move_batch_to_device(batch, self.device)

    def __len__(self) -> int:
        return len(self.data_loader)


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in batch to device."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {
            key: _move_batch_to_device(value, device) for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]
    return batch


def resolve_tcav_device(requested: str) -> torch.device:
    """Resolve device string to torch.device with fallback."""
    req = requested.strip().lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)
