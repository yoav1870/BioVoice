from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


def tensor_shape(x: Any):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
        return [tuple(t.shape) for t in x]
    return type(x).__name__


def get_time_dim(shape: tuple[int, ...] | list[tuple[int, ...]] | str):
    if isinstance(shape, tuple):
        if len(shape) == 4:
            return shape[-1]  # [B, C, F, T]
        if len(shape) == 3:
            return shape[-1]  # [B, C, T] or [B, F, T]
    return None


def inspect_model(
    model_name: str = "b2", train_type: str = "ptn", dataset: str = "vox2"
):
    model = torch.hub.load(
        "IDRnD/ReDimNet",
        "ReDimNet",
        model_name=model_name,
        train_type=train_type,
        dataset=dataset,
    )
    model.eval()

    x = torch.zeros(1, 16000)
    traces: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    handles = []

    def add_hook(name: str, module):
        def hook(_module, inputs, output):
            in_shape = tensor_shape(inputs[0] if inputs else None)
            out_shape = tensor_shape(output)
            traces[name] = {
                "module": type(_module).__name__,
                "in_shape": in_shape,
                "out_shape": out_shape,
                "t_in": get_time_dim(in_shape),
                "t_out": get_time_dim(out_shape),
            }

        handles.append(module.register_forward_hook(hook))

    add_hook("spec", model.spec)
    add_hook("backbone.stem", model.backbone.stem)
    add_hook("backbone.stage0", model.backbone.stage0)
    add_hook("backbone.stage1", model.backbone.stage1)
    add_hook("backbone.stage2", model.backbone.stage2)
    add_hook("backbone.stage3", model.backbone.stage3)
    add_hook("backbone.stage4", model.backbone.stage4)
    add_hook("backbone.stage5", model.backbone.stage5)
    add_hook("backbone.fin_wght1d", model.backbone.fin_wght1d)
    add_hook("backbone.mfa", model.backbone.mfa)
    add_hook("backbone.fin_to2d", model.backbone.fin_to2d)
    add_hook("pool", model.pool)
    add_hook("bn", model.bn)
    add_hook("linear", model.linear)

    with torch.no_grad():
        _ = model(x)

    for handle in handles:
        handle.remove()

    print(f"MODEL: {model_name}")
    print("=== Time-Dimension Trace ===")
    for name, info in traces.items():
        print(
            f"{name:22s} | {info['module']:20s} | "
            f"in={info['in_shape']} -> out={info['out_shape']} | "
            f"T: {info['t_in']} -> {info['t_out']}"
        )

    spec_t = traces["spec"]["t_out"]
    stage_t = {
        name: info["t_out"]
        for name, info in traces.items()
        if name.startswith("backbone.stage")
    }
    unique_stage_t = sorted(set(stage_t.values()))

    print("\n=== Summary ===")
    print(f"Spec output T = {spec_t}")
    print("Backbone stage T values:")
    for name, t in stage_t.items():
        print(f"  {name}: T={t}")
    print(f"Unique stage T values: {unique_stage_t}")

    if len(unique_stage_t) == 1 and spec_t == unique_stage_t[0]:
        print(
            "\nEffective time downsampling factor inside the backbone: 1x "
            "(time length preserved across stages)."
        )
    elif spec_t and unique_stage_t and unique_stage_t[0]:
        print(
            "\nEffective time downsampling factor is not 1x. "
            f"Compare spec T={spec_t} to later T={unique_stage_t}."
        )


if __name__ == "__main__":
    inspect_model(model_name="b6")
