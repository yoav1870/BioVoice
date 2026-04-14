"""Config-driven experiment runner with full reproducibility.

Provides ExperimentRunner -- a context-manager interface for running experiments
with deterministic seeding, artifact storage, and model loading.

Usage:
    runner = ExperimentRunner("config/experiment.yaml")
    with runner.run() as exp_dir:
        model = runner.load_model()
        # ... do experiment ...
        runner.save_results({"eer": 0.05})
"""

import datetime
import os
from contextlib import contextmanager

import torch
import yaml

from .reproducibility import create_experiment_dir, set_seed


class ExperimentRunner:
    """Config-driven experiment runner with reproducibility guarantees.

    Wraps experiment execution with:
        - Deterministic seeding (all 7 randomness sources)
        - Timestamped results directory with config/git/env/seed artifacts
        - Model loading via load_rawnet2()
        - Results saving with completion marker

    Attributes:
        config: Parsed experiment config dict.
        xai_root: Absolute path to xai/ module root.
        exp_dir: Path to current experiment directory (set during run()).
    """

    def __init__(self, config_path: str, xai_root: str = None):
        """Initialize runner from a YAML config file.

        Args:
            config_path: Path to experiment YAML config. If relative,
                resolved against xai_root.
            xai_root: Root directory of xai/ module. If None, auto-detected
                as the parent of the experiments/ package.
        """
        if xai_root is None:
            xai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.xai_root = os.path.abspath(xai_root)

        # Resolve config path
        if not os.path.isabs(config_path):
            config_path = os.path.join(self.xai_root, config_path)

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.exp_dir = None

    @contextmanager
    def run(self):
        """Context manager for a reproducible experiment run.

        Sets the seed, creates the experiment directory with all artifacts,
        yields the directory path, and prints a summary on exit.

        Yields:
            str: Path to the experiment results directory.
        """
        # Set deterministic seed
        seed = self.config.get("experiment", {}).get("seed", 42)
        set_seed(seed)

        # Create experiment directory with all artifacts
        self.exp_dir = create_experiment_dir(self.config)

        try:
            yield self.exp_dir
        finally:
            # Print experiment summary
            exp_name = self.config.get("experiment", {}).get("name", "unnamed")
            print(f"\n=== Experiment Complete: {exp_name} ===")
            print(f"  Results dir: {self.exp_dir}")
            completed_marker = os.path.join(self.exp_dir, "completed.txt")
            if os.path.isfile(completed_marker):
                print("  Status: COMPLETED")
            else:
                print("  Status: INCOMPLETE (no results saved)")
            print("=" * 50)

    def load_model(self) -> torch.nn.Module:
        """Load RawNet2 model using config settings.

        Uses config["model"] section to resolve config_path and weights_path
        relative to xai_root, then calls load_rawnet2().

        Returns:
            RawNet2 model in eval mode on the configured device.

        Raises:
            RuntimeError: If called outside a run() context.
        """
        from models.loader import load_rawnet2

        model_config = self.config.get("model", {})
        config_path = model_config.get("config_path", "config/model_config_RawNet.yaml")
        weights_path = model_config.get("weights_path", "models/weights/pre_trained_DF_RawNet2.pth")
        device = self.config.get("experiment", {}).get("device", "cuda:0")

        # Resolve relative paths against xai_root
        if not os.path.isabs(config_path):
            config_path = os.path.join(self.xai_root, config_path)
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(self.xai_root, weights_path)

        return load_rawnet2(config_path, weights_path, device=device, verbose=False)

    def save_results(self, results: dict, filename: str = "results.yaml") -> None:
        """Save experiment results to the current experiment directory.

        Also creates a completed.txt marker file with timestamp.

        Args:
            results: Dictionary of results to save as YAML.
            filename: Output filename (default: results.yaml).

        Raises:
            RuntimeError: If called outside a run() context.
        """
        if self.exp_dir is None:
            raise RuntimeError("save_results() must be called within a run() context")

        # Save results
        results_path = os.path.join(self.exp_dir, filename)
        with open(results_path, "w") as f:
            yaml.dump(results, f, default_flow_style=False)

        # Save completion marker
        completed_path = os.path.join(self.exp_dir, "completed.txt")
        with open(completed_path, "w") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Completed at {timestamp}\n")
