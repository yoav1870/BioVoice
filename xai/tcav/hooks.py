"""Activation extraction hooks for RawNet2 TCAV pipeline.

Provides ActivationExtractor context manager that registers forward hooks on
RawNet2 named modules and captures intermediate activations for CAV training
or TCAV sensitivity scoring.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


# Maps friendly layer names to RawNet2 module attribute names (verified Phase 01)
LAYER_NAMES: Dict[str, str] = {
    'sinc_conv':   'Sinc_conv',
    'resblock_g1': 'block1',
    'resblock_g2': 'block5',
    'pre_gru':     'bn_before_gru',
    'post_gru':    'gru',
}


def pool_activation(act: torch.Tensor) -> torch.Tensor:
    """Pool variable-length activation tensor to a fixed-size vector per sample.

    Reduces variable-length activations to fixed-size vectors for CAV training
    (see 03-RESEARCH.md Pitfall 2).

    Rules:
    - 3D (batch, C, T) where C < T  -> Global Average Pool over T axis (conv layers)
    - 3D (batch, seq_len, H) where seq_len < H -> Last timestep (GRU layers)
    - 2D (batch, D) -> return as-is

    Args:
        act: Activation tensor from a RawNet2 layer.

    Returns:
        Pooled tensor of shape (batch, features).
    """
    if act.dim() == 3:
        batch, d1, d2 = act.shape
        if d1 < d2 and d2 > d1 * 8:
            # (batch, seq_len, H) where H >> seq_len -- GRU hidden state: last timestep
            # Discriminator: H is substantially larger than seq_len (e.g. 1024 vs 10-20)
            return act[:, -1, :]
        elif d1 <= d2:
            # (batch, C, T) where C < T -- conv layer: Global Average Pool over time
            return act.mean(dim=-1)
        else:
            # (batch, T, C) where T > C -- transpose conv: Global Average Pool over time
            return act.mean(dim=1)
    elif act.dim() == 2:
        return act
    else:
        raise ValueError(f"Unexpected activation dimensionality: {act.dim()}D shape {act.shape}")


class ActivationExtractor:
    """Context manager that registers forward hooks on RawNet2 layers to capture activations.

    Two modes controlled by retain_grad parameter:
    - retain_grad=False (default): For CAV training. Detaches activation from graph
      and moves to CPU. Use inside torch.no_grad() for efficiency.
    - retain_grad=True: For TCAV sensitivity scoring. Keeps activation in autograd
      graph and calls retain_grad() so gradients can be computed w.r.t. activations.
      Do NOT use with torch.no_grad() in this mode.

    Args:
        model: RawNet2 torch.nn.Module.
        layer_names: Dict mapping friendly key -> module attribute name (e.g. LAYER_NAMES).
        retain_grad: If True, keep activations in autograd graph for TCAV sensitivity scoring.

    Example:
        with ActivationExtractor(model, LAYER_NAMES) as ext:
            with torch.no_grad():
                _ = model(audio)
            act = ext.get('sinc_conv')  # shape (batch, C)
    """

    def __init__(self, model: nn.Module, layer_names: Dict[str, str],
                 retain_grad: bool = False):
        self.model = model
        self.layer_names = layer_names
        self.retain_grad = retain_grad
        self.activations: Dict[str, Optional[torch.Tensor]] = {}
        self._handles = []

    def _save(self, key: str):
        """Create hook callback that saves the activation for the given key."""
        def hook(module, input, output):
            # Unpack GRU tuple output: GRU returns (output, h_n) or (output, (h_n, c_n))
            if isinstance(output, tuple):
                output = output[0]
            if self.retain_grad:
                # TCAV scoring path: keep in autograd graph, enable gradient accumulation
                output.retain_grad()
                self.activations[key] = output
            else:
                # CAV training path: detach from graph and move to CPU
                self.activations[key] = output.detach().cpu()
        return hook

    def __enter__(self):
        self.activations = {}
        self._handles = []
        # Build module lookup by attribute name
        named_modules = dict(self.model.named_modules())
        for key, attr in self.layer_names.items():
            if attr not in named_modules:
                raise KeyError(
                    f"Module '{attr}' (key='{key}') not found in model. "
                    f"Available modules: {list(named_modules.keys())[:10]}..."
                )
            module = named_modules[attr]
            handle = module.register_forward_hook(self._save(key))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always remove hooks to prevent memory leaks (T-03-03 mitigation)
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False  # Do not suppress exceptions

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return the stored activation tensor for the given layer key.

        Args:
            key: Friendly layer name (e.g. 'sinc_conv', 'post_gru').

        Returns:
            Stored activation tensor, or None if not yet captured.
        """
        return self.activations.get(key, None)
