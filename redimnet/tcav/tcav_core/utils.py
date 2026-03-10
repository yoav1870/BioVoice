from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def abs_path(path: Path) -> Path:
    return path if path.is_absolute() else (project_root() / path)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**32)


def output_path(output_dir: Path, output_tag: str, name: str) -> Path:
    tag = output_tag.strip()
    filename = f"{tag}-{name}" if tag else name
    return output_dir / filename
