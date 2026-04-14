"""Reproducibility utilities: deterministic seeding and experiment artifact storage.

Provides:
    set_seed(seed) -- full deterministic seeding covering 7 randomness sources
    create_experiment_dir(config, base_dir) -- timestamped results dir with config/git/env/seed artifacts
    verify_experiment_dir(exp_dir) -- check that all expected artifacts exist
"""

import datetime
import os
import random
import subprocess
import sys

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set seed for full deterministic reproducibility.

    Covers all 7 randomness sources:
        1. random.seed
        2. np.random.seed
        3. torch.manual_seed
        4. torch.cuda.manual_seed_all
        5. os.environ[PYTHONHASHSEED]
        6. torch.backends.cudnn.deterministic
        7. torch.backends.cudnn.benchmark

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed} (deterministic mode enabled)")


def create_experiment_dir(config: dict, base_dir: str = None) -> str:
    """Create a timestamped experiment directory with full reproducibility artifacts.

    Saves 4 artifact files:
        - config_snapshot.yaml: full config dict
        - git_hash.txt: current git commit hash (or "not-a-git-repo")
        - environment.txt: pip freeze output
        - seed.txt: seed value from config

    Args:
        config: Experiment configuration dictionary. Must contain
            config["experiment"]["name"] and config["experiment"]["seed"].
        base_dir: Base directory for experiment output. If None, uses
            config["output"]["results_dir"] resolved relative to xai/ root.

    Returns:
        Path to the created experiment directory.
    """
    # Resolve base_dir
    if base_dir is None:
        xai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = config.get("output", {}).get("results_dir", "results/")
        base_dir = os.path.join(xai_root, results_dir)

    # Build timestamped directory name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    exp_dir = os.path.join(base_dir, f"{timestamp}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save config snapshot
    with open(os.path.join(exp_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save git hash
    try:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "not-a-git-repo"
    with open(os.path.join(exp_dir, "git_hash.txt"), "w") as f:
        f.write(git_hash)

    # Save pip freeze (use the same Python interpreter that is running)
    try:
        pip_freeze = (
            subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pip_freeze = "pip freeze failed"
    with open(os.path.join(exp_dir, "environment.txt"), "w") as f:
        f.write(pip_freeze)

    # Save seed
    seed = config.get("experiment", {}).get("seed", "unknown")
    with open(os.path.join(exp_dir, "seed.txt"), "w") as f:
        f.write(str(seed))

    return exp_dir


def verify_experiment_dir(exp_dir: str) -> dict:
    """Verify that an experiment directory contains all required artifacts.

    Args:
        exp_dir: Path to experiment directory to verify.

    Returns:
        Dictionary with:
            complete (bool): True if all artifacts present.
            missing (list[str]): Names of missing artifact files.
            artifacts (dict): Maps filename to first line of content.
    """
    required = ["config_snapshot.yaml", "git_hash.txt", "environment.txt", "seed.txt"]
    missing = []
    artifacts = {}

    for filename in required:
        filepath = os.path.join(exp_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                first_line = f.readline().strip()
            artifacts[filename] = first_line
        else:
            missing.append(filename)

    return {
        "complete": len(missing) == 0,
        "missing": missing,
        "artifacts": artifacts,
    }
