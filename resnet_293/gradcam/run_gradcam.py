#!/usr/bin/env python3
"""
GradCAM analysis on ResNet293 for ASVspoof5.

For a selected system, loads utterances from tar archives, runs GradCAM
on the chosen layer, and saves heatmap visualisations showing which
time-frequency regions the model focuses on for spoof detection.

Usage:
    cd /home/SpeakerRec/BioVoice/resnet_293/gradcam
    /home/SpeakerRec/BioVoice/.venv/bin/python3 run_gradcam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---- bootstrap imports from the tcav package --------------------------------
from config import (
    BIOVOICE_ROOT,
    GRADCAM_LAYER,
    MAX_SPEAKERS_PER_CLASS,
    MAX_UTTS_PER_SPEAKER,
    OUTPUT_DIR,
    TARGET_SYSTEM,
    TCAV_DIR,
    THRESHOLD,
)

sys.path.insert(0, str(TCAV_DIR))

import matplotlib

matplotlib.use("Agg")  # headless backend for the server

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam

from captum_tcav.asvspoof5.config import load_config
from captum_tcav.asvspoof5.data import (
    build_tar_index,
    load_input,
    load_manifest,
    partition_for_system,
    select_subset_for_class,
    shared_speakers_for_both_classes,
    validate_config,
)
from captum_tcav.asvspoof5.modeling import load_model

# ---- helpers ----------------------------------------------------------------


def compute_gradcam(
    gradcam: LayerGradCam,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """Run GradCAM and return a normalised heatmap in (F, T) space.

    The model internally permutes (B,T,F) -> (B,1,F,T), so the layer-4
    feature map has spatial dims (F, T).  We upscale to (F=80, T=200)
    so it lines up with the fbank displayed as (freq-y, time-x).
    """
    input_tensor = input_tensor.detach().clone().requires_grad_(True)
    # target=0 -> gradient w.r.t. the spoof logit (only output column)
    attribution = gradcam.attribute(input_tensor, target=0)
    # attribution shape: (B, 1, H, W)  where H~freq, W~time
    # upscale to match the internal (F, T) = (80, 200) layout
    heatmap = (
        F.interpolate(
            attribution,
            size=(80, 200),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )  # (80, 200)

    # normalise to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-8:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap  # shape (80, 200)


def save_single_plot(
    fbank_tf: np.ndarray,
    heatmap_ft: np.ndarray,
    threshold: float,
    label: str,
    utt_id: str,
    speaker_id: str,
    system_id: str,
    out_path: Path,
) -> None:
    """Save a 3-panel figure: fbank | GradCAM overlay | thresholded overlay."""
    fbank_ft = fbank_tf.T  # (T,F) -> (F,T) for imshow
    heatmap_thresh = np.where(heatmap_ft >= threshold, heatmap_ft, 0.0)
    pct_above = float((heatmap_ft >= threshold).mean() * 100)

    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    # 1  raw fbank
    axes[0].imshow(fbank_ft, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title(f"Fbank  ({label})\n{speaker_id} / {utt_id}", fontsize=9)
    axes[0].set_xlabel("Time frame")
    axes[0].set_ylabel("Mel bin")

    # 2  GradCAM overlay
    axes[1].imshow(fbank_ft, aspect="auto", origin="lower", cmap="magma", alpha=0.5)
    im = axes[1].imshow(
        heatmap_ft, aspect="auto", origin="lower", cmap="jet", alpha=0.5
    )
    axes[1].set_title(f"GradCAM  (layer={GRADCAM_LAYER})", fontsize=9)
    axes[1].set_xlabel("Time frame")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3  thresholded overlay
    axes[2].imshow(fbank_ft, aspect="auto", origin="lower", cmap="magma", alpha=0.5)
    im2 = axes[2].imshow(
        heatmap_thresh,
        aspect="auto",
        origin="lower",
        cmap="jet",
        alpha=0.6,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(f"Threshold >= {threshold}\n{pct_above:.1f}% above", fontsize=9)
    axes[2].set_xlabel("Time frame")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"{system_id}  {label}  {speaker_id}  {utt_id}", fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---- main -------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys_out = OUTPUT_DIR / TARGET_SYSTEM
    sys_out.mkdir(parents=True, exist_ok=True)

    config = load_config()
    validate_config(config)

    print(f"System:    {TARGET_SYSTEM}")
    print(f"Layer:     {GRADCAM_LAYER}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Output:    {sys_out}")
    print()

    # ---- load model ---------------------------------------------------------
    print("Loading model …")
    model = load_model(config, TARGET_SYSTEM)
    model.eval()
    target_layer = getattr(model, GRADCAM_LAYER)
    gradcam = LayerGradCam(model, target_layer)
    print("Model loaded (CPU).\n")

    # ---- load data ----------------------------------------------------------
    print("Building tar index …")
    manifest = load_manifest(config, TARGET_SYSTEM)
    tar_index = build_tar_index(config, TARGET_SYSTEM)

    shared_speakers = shared_speakers_for_both_classes(config, TARGET_SYSTEM, manifest)
    print(f"Shared speakers: {len(shared_speakers)}\n")

    # ---- collect per-class stats for summary --------------------------------
    class_heatmaps: dict[str, list[np.ndarray]] = {"spoof": [], "bonafide": []}
    class_pct_above: dict[str, list[float]] = {"spoof": [], "bonafide": []}

    for target_class in ("spoof", "bonafide"):
        subset = select_subset_for_class(
            config,
            TARGET_SYSTEM,
            manifest,
            target_class,
            forced_speakers=shared_speakers,
        )
        speakers = subset.selected_speakers[:MAX_SPEAKERS_PER_CLASS]
        rows = subset.selected_rows

        for speaker_id in speakers:
            speaker_df = rows[rows["speaker_id"] == speaker_id].head(
                MAX_UTTS_PER_SPEAKER
            )
            utt_ids = speaker_df["utt_id"].astype(str).tolist()

            for utt_id in utt_ids:
                input_tensor = load_input(
                    config, model, [utt_id], tar_index
                )  # (1, T, F)
                fbank_tf = input_tensor.squeeze().detach().cpu().numpy()  # (T, F)

                heatmap_ft = compute_gradcam(gradcam, input_tensor)  # (F, T)
                pct = float((heatmap_ft >= THRESHOLD).mean() * 100)

                class_heatmaps[target_class].append(heatmap_ft)
                class_pct_above[target_class].append(pct)

                out_path = sys_out / f"{target_class}_{speaker_id}_{utt_id}.png"
                save_single_plot(
                    fbank_tf,
                    heatmap_ft,
                    THRESHOLD,
                    target_class,
                    utt_id,
                    speaker_id,
                    TARGET_SYSTEM,
                    out_path,
                )
                print(
                    f"  [{target_class:>8s}] {speaker_id} {utt_id}  above_thr={pct:5.1f}%"
                )

    # ---- summary plot -------------------------------------------------------
    print("\nGenerating summary …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cls in zip(axes, ("bonafide", "spoof")):
        maps = class_heatmaps[cls]
        if maps:
            avg_map = np.mean(np.stack(maps, axis=0), axis=0)
            avg_pct = np.mean(class_pct_above[cls])
        else:
            avg_map = np.zeros((80, 200))
            avg_pct = 0.0
        im = ax.imshow(
            avg_map, aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=1
        )
        ax.set_title(
            f"{cls}  (n={len(maps)})\navg above {THRESHOLD}: {avg_pct:.1f}%",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Mel bin")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Average GradCAM Heatmap — {TARGET_SYSTEM} — {GRADCAM_LAYER}\n"
        f"threshold={THRESHOLD}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    summary_path = sys_out / "summary_avg_heatmap.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary saved: {summary_path}")

    # ---- final stats --------------------------------------------------------
    print("\n===== Results =====")
    for cls in ("bonafide", "spoof"):
        vals = class_pct_above[cls]
        if vals:
            print(
                f"  {cls:>8s}: n={len(vals):3d}  "
                f"avg_above_thr={np.mean(vals):5.1f}%  "
                f"min={np.min(vals):5.1f}%  max={np.max(vals):5.1f}%"
            )
    print("\nDone.")


if __name__ == "__main__":
    main()
