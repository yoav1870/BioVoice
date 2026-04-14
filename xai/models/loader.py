"""Model loading utility for RawNet2."""

import os
import torch
import yaml

from .rawnet2 import RawNet


def load_rawnet2(
    config_path: str,
    weights_path: str,
    device: str = "cuda:0",
    verbose: bool = True,
) -> torch.nn.Module:
    """Load pretrained RawNet2 with weight verification.

    Args:
        config_path: Path to model_config_RawNet.yaml (absolute or relative to xai/).
        weights_path: Path to pretrained .pth weights file (absolute or relative to xai/).
        device: Target device (e.g., "cuda:0" or "cpu").
        verbose: If True, print architecture summary.

    Returns:
        RawNet2 model in eval mode on the specified device.
    """
    # Resolve paths relative to xai/ directory if not absolute
    xai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(config_path):
        config_path = os.path.join(xai_root, config_path)
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(xai_root, weights_path)

    # Validate files exist
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Load model config
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)["model"]

    # Instantiate model
    model = RawNet(model_config, device)

    # Load pretrained weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()

    if verbose:
        print("=== RawNet2 Architecture ===")
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {device}")
        print(f"  Weights: {os.path.basename(weights_path)}")
        print("  Key layers:")
        for name, module in model.named_modules():
            if name in (
                "Sinc_conv", "first_bn",
                "block0", "block1", "block2", "block3", "block4", "block5",
                "bn_before_gru", "gru", "fc1_gru", "fc2_gru",
            ):
                print(f"    {name}: {module.__class__.__name__}")
        print("=== Model loaded successfully ===")

    return model
