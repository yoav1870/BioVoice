"""TCAV sensitivity scorer for RawNet2.

Computes TCAV scores (fraction of positive directional derivatives) for one
(class, concept, layer) triple using autograd. Directional derivatives are
computed via forward+backward pass with retain_grad() on target layer activation.

CRITICAL: model.eval() only -- NO torch.no_grad() in the scoring path.
Gradients must flow through the model for directional derivative computation.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from xai.tcav.hooks import LAYER_NAMES
from typing import Dict


def compute_tcav_score(model: torch.nn.Module, inputs: torch.Tensor,
                       target_class: int, layer_name: str,
                       cav: np.ndarray, scaler,
                       device: str = 'cuda:0',
                       batch_size: int = 32) -> Dict:
    """
    Compute TCAV score for one (class, concept, layer) triple.

    TCAV score = fraction of examples where directional derivative
    (grad of logit w.r.t. layer activation, dotted with CAV) is positive.

    model:        RawNet2 in eval mode (do NOT wrap in torch.no_grad())
    inputs:       (N, 64600) raw waveform tensor -- all examples for one class
    target_class: 0=spoof, 1=bonafide (Phase 1: output[:,1]=bonafide)
    layer_name:   friendly name from LAYER_NAMES (e.g. 'sinc_conv')
    cav:          unit-normalized CAV vector from train_cav(), shape (d,)
    scaler:       fitted StandardScaler from train_cav() -- NOT USED for gradient
                  computation but passed for API consistency
    device:       CUDA device string
    batch_size:   batch size for forward passes

    Returns: {'tcav_score': float in [0,1], 'directional_derivs': np.ndarray}
    """
    model.train()  # train mode required for CuDNN GRU backward; eval() blocks RNN gradients
    cav_tensor = torch.tensor(cav, dtype=torch.float32, device=device)
    derivs = []

    dataset = TensorDataset(inputs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for (batch,) in loader:
        x = batch.to(device).requires_grad_(True)  # requires_grad essential for SincConv

        # Register forward hook with retain_grad on target layer
        attr_name = LAYER_NAMES[layer_name]
        module = dict(model.named_modules())[attr_name]
        layer_acts = {}

        def _hook(m, inp, out, key=layer_name):
            o = out[0] if isinstance(out, tuple) else out
            o.retain_grad()
            layer_acts[key] = o

        handle = module.register_forward_hook(_hook)

        # Forward pass (no torch.no_grad()!)
        output = model(x)
        handle.remove()  # remove immediately after forward

        # Handle tuple output: RawNet2 returns logits directly, but check
        if isinstance(output, tuple):
            output = output[1]

        # Backpropagate from target class logit
        logit = output[:, target_class]
        logit.sum().backward()

        # Get gradient at target layer
        act = layer_acts[layer_name]
        grad = act.grad
        if grad is None:
            raise RuntimeError(
                f"Gradient is None at layer {layer_name} -- "
                f"check input.requires_grad_(True) and no torch.no_grad() wrapper"
            )

        # Pool gradient same way as activation was pooled for CAV training
        # Then dot with CAV
        grad_np = grad.detach().cpu().numpy()
        # Flatten spatial dims: (batch, *spatial) -> (batch, d)
        grad_flat = grad_np.reshape(len(batch), -1)

        # If activation was pooled (GAP for conv, last timestep for GRU),
        # the CAV was trained on pooled activations. The gradient w.r.t.
        # the unpooled activation needs to be pooled the same way.
        # For conv (batch, C, T): mean over T gives (batch, C)
        # For GRU (batch, seq, H): last timestep gives (batch, H)
        if grad.dim() == 3:
            C, T = grad.shape[1], grad.shape[2]
            if C < T and T > C * 8:  # GRU: last timestep (seq_len << H)
                grad_pooled = grad.detach().cpu().numpy()[:, -1, :]  # (batch, H)
            elif C <= T:  # conv: GAP over time
                grad_pooled = grad.detach().cpu().numpy().mean(axis=-1)  # (batch, C)
            else:
                grad_pooled = grad_flat
        elif grad.dim() == 2:
            grad_pooled = grad.detach().cpu().numpy()  # already (batch, D)
        else:
            grad_pooled = grad_flat

        deriv = grad_pooled @ cav  # (batch,) directional derivative per example
        derivs.extend(deriv.tolist())

        # Clear grads for next batch
        model.zero_grad()

    derivs = np.array(derivs)
    return {
        'tcav_score': float(np.mean(derivs > 0)),
        'directional_derivs': derivs,
    }
