"""Gradient flow verification for RawNet2 target layers.

Verifies that gradients propagate through all target layer groups:
SincConv, ResBlock groups 1 and 2, GRU, and post-GRU FC layers.

FOUND-02: Gradient flow must be confirmed before TCAV analysis.

Note: SincConv has no learnable nn.Parameter (its filters are
reconstructed from numpy arrays each forward pass). We verify
SincConv gradient flow via a forward hook that captures the output
activation and calls retain_grad() so we can inspect its gradient
after backward. For this to work, the input tensor must have
requires_grad=True so the SincConv output is part of the autograd
graph. This is the same requirement TCAV has (it needs gradients
of the loss w.r.t. intermediate activations).
"""

import torch
import torch.nn as nn


# Map friendly names to model attribute prefixes
# These match the RawNet class attribute names in rawnet2.py
TARGET_LAYERS = {
    "SincConv": "Sinc_conv",
    "ResBlock_group1_block0": "block0",
    "ResBlock_group1_block1": "block1",
    "ResBlock_group2_block2": "block2",
    "ResBlock_group2_block3": "block3",
    "ResBlock_group2_block4": "block4",
    "ResBlock_group2_block5": "block5",
    "GRU": "gru",
    "FC1_post_GRU": "fc1_gru",
    "FC2_post_GRU": "fc2_gru",
}

# Layers that have no nn.Parameter and must be checked via activation hooks.
# For these layers, we register a forward hook that captures the output and
# calls retain_grad(), then check the gradient after backward.
_ACTIVATION_ONLY_LAYERS = {"Sinc_conv"}


def _get_module_by_name(model, name):
    """Get a submodule by its attribute name (e.g., 'Sinc_conv', 'block0')."""
    for n, m in model.named_modules():
        if n == name:
            return m
    return None


def verify_gradient_flow(model, sample_input, device="cuda:0"):
    """Run forward+backward pass and verify gradients at all target layers.

    For layers with learnable parameters (ResBlocks, GRU, FC), checks that
    param.grad is not None and has non-zero mean absolute value.

    For SincConv (no learnable parameters), registers a forward hook that
    captures the output activation tensor and calls retain_grad(). After
    backward, checks whether the activation has a non-zero gradient. This
    mirrors the TCAV workflow which also requires activation gradients.

    The input tensor is given requires_grad=True so that the SincConv
    output (produced by F.conv1d with fixed filters applied to the input)
    participates in the autograd graph. This is the same requirement as
    TCAV concept sensitivity analysis.

    Args:
        model: RawNet2 model instance (from load_rawnet2).
        sample_input: Tensor of shape (batch, 64600).
        device: CUDA device string.

    Returns:
        dict mapping friendly_name -> {
            has_grad: bool,
            mean_abs: float,
            max_abs: float,
            num_params: int,
            method: str ('parameter' or 'activation'),
            details: str (diagnostic info if no gradient)
        }
    """
    original_training = model.training

    # Set train mode for gradient computation
    model.train()
    model.zero_grad()

    # Storage for activation-based gradient checks (forward hook approach)
    activation_outputs = {}  # friendly_name -> captured output tensor
    hooks = []

    # Register forward hooks for activation-only layers (SincConv)
    for friendly_name, attr_name in TARGET_LAYERS.items():
        if attr_name in _ACTIVATION_ONLY_LAYERS:
            module = _get_module_by_name(model, attr_name)
            if module is not None:

                def _make_fwd_hook(fname):
                    def hook_fn(module, input, output):
                        # Capture the output and retain its gradient
                        if output.requires_grad:
                            output.retain_grad()
                        activation_outputs[fname] = output

                    return hook_fn

                h = module.register_forward_hook(_make_fwd_hook(friendly_name))
                hooks.append(h)

    # Prepare input: requires_grad=True so SincConv output is in autograd graph
    # (SincConv uses F.conv1d with fixed numpy-derived filters; without
    # requires_grad on the input, the output won't track gradients)
    inp = sample_input.to(device).requires_grad_(True)

    # Enable anomaly detection to catch in-place op issues
    with torch.autograd.set_detect_anomaly(True):
        # Forward pass
        output = model(inp)

        # Handle tuple output (some RawNet variants return (embedding, logits))
        if isinstance(output, tuple):
            output = output[-1]  # take logits

        # Backward: bonafide class logit sum (for TCAV-style gradient analysis)
        loss = output[:, 1].sum()
        loss.backward()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Collect results
    results = {}

    for friendly_name, attr_name in TARGET_LAYERS.items():
        # Check if this was handled by activation forward hook
        if friendly_name in activation_outputs:
            act = activation_outputs[friendly_name]
            if act.grad is not None:
                results[friendly_name] = {
                    "has_grad": True,
                    "mean_abs": act.grad.abs().mean().item(),
                    "max_abs": act.grad.abs().max().item(),
                    "num_params": 0,
                    "method": "activation",
                    "details": (
                        "Gradient measured at module output activation "
                        "(no learnable params; fixed Mel-scale filters)"
                    ),
                }
            else:
                results[friendly_name] = {
                    "has_grad": False,
                    "mean_abs": 0.0,
                    "max_abs": 0.0,
                    "num_params": 0,
                    "method": "activation",
                    "details": (
                        "Activation gradient is None -- input may not require grad, "
                        "or SincConv output is not in the autograd graph"
                    ),
                }
            continue

        # Parameter-based gradient check
        matching_params = []
        for pname, param in model.named_parameters():
            if pname.startswith(attr_name + "."):
                matching_params.append((pname, param))

        if not matching_params:
            results[friendly_name] = {
                "has_grad": False,
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "num_params": 0,
                "method": "parameter",
                "details": "No parameters found with prefix '{}'".format(attr_name),
            }
            continue

        # Aggregate gradient info across all matching parameters
        grad_means = []
        grad_maxes = []
        has_any_grad = False
        none_params = []
        zero_params = []

        for pname, param in matching_params:
            if param.grad is not None:
                mean_val = param.grad.abs().mean().item()
                max_val = param.grad.abs().max().item()
                if mean_val > 0:
                    has_any_grad = True
                    grad_means.append(mean_val)
                    grad_maxes.append(max_val)
                else:
                    zero_params.append(pname)
            else:
                none_params.append(pname)

        # A layer passes if at least one parameter has non-zero gradient.
        # BatchNorm weight/bias can have zero grad with random input in train
        # mode -- this is OK as long as the conv layers in the same block have
        # gradients flowing through them.
        overall_has_grad = has_any_grad
        overall_mean = sum(grad_means) / len(grad_means) if grad_means else 0.0
        overall_max = max(grad_maxes) if grad_maxes else 0.0

        details = ""
        if none_params:
            details += "Params with grad=None: {}. ".format(none_params)
        if zero_params:
            details += "Params with grad=0: {}. ".format(zero_params)
        if not details:
            details = "All {} params have non-zero gradients".format(len(matching_params))

        results[friendly_name] = {
            "has_grad": overall_has_grad,
            "mean_abs": overall_mean,
            "max_abs": overall_max,
            "num_params": len(matching_params),
            "method": "parameter",
            "details": details,
        }

    # Restore original mode
    if not original_training:
        model.eval()

    return results
