from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ace_pipeline_v2 import (  # type: ignore
    REPO_ROOT,
    LayerEmbedder,
    _bbox_iou,
    _choose_k_auto,
    _cluster_per_scale,
    _compute_gradcam_ft,
    _connected_components,
    _embed_regions,
    _expand_to_min,
    _scale_name,
    _set_seed,
    _zscore_patch,
)

from captum_tcav.asvspoof5.config import load_config  # type: ignore
from captum_tcav.asvspoof5.data import (  # type: ignore
    build_tar_index,
    load_input,
    load_manifest,
    select_subset_for_class,
    shared_speakers_for_both_classes,
    validate_config,
)
from captum_tcav.asvspoof5.modeling import load_model  # type: ignore


@dataclass
class RegionItemV3:
    utt_id: str
    speaker_id: str
    target_class: str
    t0: int
    t1: int
    f0: int
    f1: int
    score: float
    scale_name: str
    arr_tf: np.ndarray  # masked patch (T,F)
    mask_tf: np.ndarray  # binary mask (T,F)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _binary_dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(iters))):
        p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        neigh = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
        out = neigh
    return out


def _binary_erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(iters))):
        p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        neigh = (
            p[1:-1, 1:-1]
            & p[:-2, 1:-1]
            & p[2:, 1:-1]
            & p[1:-1, :-2]
            & p[1:-1, 2:]
            & p[:-2, :-2]
            & p[:-2, 2:]
            & p[2:, :-2]
            & p[2:, 2:]
        )
        out = neigh
    return out


def _binary_open(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return _binary_dilate(_binary_erode(mask, iters=iters), iters=iters)


def _binary_close(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, iters=iters), iters=iters)


def _extract_regions_from_one_utt_v3(
    cfg: dict[str, Any],
    fbank_tf: np.ndarray,  # (T,F)
    hm_ft: np.ndarray,  # (F,T)
) -> list[tuple[int, int, int, int, float, np.ndarray, np.ndarray]]:
    q_list = [float(q) for q in cfg["gradcam_quantiles"]]
    min_area = int(cfg["component_min_area"])
    min_t = int(cfg["component_min_t"])
    min_f = int(cfg["component_min_f"])
    pad_t = int(cfg["bbox_pad_t"])
    pad_f = int(cfg["bbox_pad_f"])
    dedup_iou = float(cfg["dedup_iou_threshold"])
    max_regions = int(cfg["max_regions_per_utt"])
    normalize_z = bool(cfg["normalize_patch_zscore"])
    heat_weight = bool(cfg["apply_heat_weighting"])
    open_iters = int(cfg.get("morph_open_iters", 0))
    close_iters = int(cfg.get("morph_close_iters", 0))

    full_t, full_f = fbank_tf.shape
    candidates: list[tuple[int, int, int, int, float, np.ndarray, np.ndarray]] = []

    for q in q_list:
        thr = float(np.quantile(hm_ft, q))
        mask = hm_ft >= thr
        if open_iters > 0:
            mask = _binary_open(mask, iters=open_iters)
        if close_iters > 0:
            mask = _binary_close(mask, iters=close_iters)

        comps = _connected_components(mask)
        for comp in comps:
            if comp.shape[0] < min_area:
                continue
            fs = comp[:, 0]
            ts = comp[:, 1]
            f0 = int(fs.min())
            f1 = int(fs.max()) + 1
            t0 = int(ts.min())
            t1 = int(ts.max()) + 1

            t0 = max(0, t0 - pad_t)
            t1 = min(full_t, t1 + pad_t)
            f0 = max(0, f0 - pad_f)
            f1 = min(full_f, f1 + pad_f)
            t0, t1, f0, f1 = _expand_to_min(
                t0=t0, t1=t1, f0=f0, f1=f1, min_t=min_t, min_f=min_f, max_t=full_t, max_f=full_f
            )
            if t1 <= t0 or f1 <= f0:
                continue

            local_mask_ft = np.zeros((f1 - f0, t1 - t0), dtype=np.float32)
            keep = (fs >= f0) & (fs < f1) & (ts >= t0) & (ts < t1)
            if not np.any(keep):
                continue
            local_mask_ft[fs[keep] - f0, ts[keep] - t0] = 1.0
            local_mask_tf = local_mask_ft.T  # (T,F)

            patch_tf = fbank_tf[t0:t1, f0:f1].astype(np.float32)
            if heat_weight:
                local_heat_tf = hm_ft[f0:f1, t0:t1].T.astype(np.float32)
                patch_tf = patch_tf * local_mask_tf * local_heat_tf
            else:
                patch_tf = patch_tf * local_mask_tf
            if normalize_z:
                patch_tf = _zscore_patch(patch_tf)

            score = float(np.mean(hm_ft[fs, ts]))
            candidates.append((t0, t1, f0, f1, score, patch_tf, local_mask_tf.astype(np.float32)))

    candidates.sort(key=lambda x: x[4], reverse=True)
    deduped: list[tuple[int, int, int, int, float, np.ndarray, np.ndarray]] = []
    kept_boxes: list[tuple[int, int, int, int]] = []
    for c in candidates:
        box = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
        if any(_bbox_iou(box, b) >= dedup_iou for b in kept_boxes):
            continue
        deduped.append(c)
        kept_boxes.append(box)
        if len(deduped) >= max_regions:
            break
    return deduped


def _collect_regions_for_class_v3(
    cfg: dict[str, Any],
    tcav_cfg: Any,
    model: torch.nn.Module,
    tar_index: dict[str, tuple[Path, str]],
    manifest: pd.DataFrame,
    system_id: str,
    target_class: str,
    shared_speakers: list[str],
    layer_name: str,
) -> list[RegionItemV3]:
    model_device = next(model.parameters()).device
    if target_class == "spoof":
        work_cfg = replace(
            tcav_cfg,
            subset_num_speakers=int(cfg["subset_num_speakers_spoof"]),
            subset_utts_per_speaker=int(cfg["subset_utts_per_speaker_spoof"]),
            subset_min_utts_per_speaker=int(cfg["subset_min_utts_per_speaker_spoof"]),
        )
    else:
        work_cfg = replace(
            tcav_cfg,
            subset_num_speakers=int(cfg["subset_num_speakers_bonafide"]),
            subset_utts_per_speaker=int(cfg["subset_utts_per_speaker_bonafide"]),
            subset_min_utts_per_speaker=int(cfg["subset_min_utts_per_speaker_bonafide"]),
        )

    subset = select_subset_for_class(
        work_cfg,
        system_id,
        manifest,
        target_class,
        forced_speakers=shared_speakers,
    )
    rows = subset.selected_rows.copy()
    rows["utt_id"] = rows["utt_id"].astype(str)
    rows["speaker_id"] = rows["speaker_id"].astype(str)
    target_layer = getattr(model, layer_name)
    scales = list(cfg["area_scales"])
    max_total = int(cfg["max_total_regions_per_class"])
    full_t, full_f = 200, 80

    out: list[RegionItemV3] = []
    print(f"[collect-v3:{target_class}] speakers={len(subset.selected_speakers)} rows={len(rows)}")
    for i, row in rows.iterrows():
        utt_id = str(row["utt_id"])
        speaker_id = str(row["speaker_id"])
        x = load_input(work_cfg, model, [utt_id], tar_index)
        fbank_tf = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        hm_ft = _compute_gradcam_ft(
            model=model,
            target_layer=target_layer,
            x_btf=x.to(model_device),
            out_size=(80, 200),
        )
        regs = _extract_regions_from_one_utt_v3(cfg, fbank_tf=fbank_tf, hm_ft=hm_ft)
        for t0, t1, f0, f1, sc, patch_tf, mask_tf in regs:
            out.append(
                RegionItemV3(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    target_class=target_class,
                    t0=t0,
                    t1=t1,
                    f0=f0,
                    f1=f1,
                    score=float(sc),
                    scale_name=_scale_name(t0, t1, f0, f1, scales=scales, full_t=full_t, full_f=full_f),
                    arr_tf=patch_tf,
                    mask_tf=mask_tf,
                )
            )
        if (i + 1) % 8 == 0 or i == len(rows) - 1:
            print(f"  - {i + 1}/{len(rows)} utts, regions={len(out)}")
        if len(out) >= max_total:
            out = out[:max_total]
            print(f"[collect-v3:{target_class}] reached max_total_regions_per_class={max_total}")
            break
    return out


def _save_cluster_concepts_v3(
    out_dir: Path,
    spoof_items: list[RegionItemV3],
    labels: np.ndarray,
    score_df: pd.DataFrame,
    top_examples_per_cluster: int,
    save_patch_previews: bool,
) -> None:
    concepts_dir = out_dir / "concepts_auto"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    for _, row in score_df.iterrows():
        cid = int(row["cluster_id"])
        idx = np.where(labels == cid)[0]
        cdir = concepts_dir / f"cluster_{cid:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        masks_dir = cdir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        keep = idx[: min(len(idx), int(top_examples_per_cluster))]

        meta_rows: list[dict[str, Any]] = []
        for j, ii in enumerate(keep):
            it = spoof_items[int(ii)]
            arr_ft = it.arr_tf.T.astype(np.float32)
            mask_ft = it.mask_tf.T.astype(np.float32)
            np.save(cdir / f"{j:06d}.npy", arr_ft)
            np.save(masks_dir / f"{j:06d}_mask.npy", mask_ft)
            meta_rows.append(
                {
                    "file": f"{j:06d}.npy",
                    "mask_file": f"masks/{j:06d}_mask.npy",
                    "utt_id": it.utt_id,
                    "speaker_id": it.speaker_id,
                    "t0": int(it.t0),
                    "t1": int(it.t1),
                    "f0": int(it.f0),
                    "f1": int(it.f1),
                    "score": float(it.score),
                    "scale_name": it.scale_name,
                }
            )

        if save_patch_previews and len(keep) > 0:
            patch_preview = [spoof_items[int(ii)].arr_tf.T.astype(np.float32) for ii in keep[:16]]
            mask_preview = [spoof_items[int(ii)].mask_tf.T.astype(np.float32) for ii in keep[:16]]
            max_f = max(x.shape[0] for x in patch_preview)
            max_t = max(x.shape[1] for x in patch_preview)
            pstack = np.zeros((len(patch_preview), max_f, max_t), dtype=np.float32)
            mstack = np.zeros((len(mask_preview), max_f, max_t), dtype=np.float32)
            for pi, (p, m) in enumerate(zip(patch_preview, mask_preview)):
                f, t = p.shape
                pstack[pi, :f, :t] = p
                mstack[pi, :f, :t] = m
            np.save(cdir / "_preview_stack.npy", pstack)
            np.save(cdir / "_preview_mask_stack.npy", mstack)

        with (cdir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "cluster_id": cid,
                    "cluster_size": int(len(idx)),
                    "saved_examples": int(len(keep)),
                    "rows": meta_rows,
                },
                f,
                indent=2,
            )


def run_ace_v3(config_path: Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    _set_seed(int(cfg["seed"]))

    tcav_cfg = load_config()
    validate_config(tcav_cfg)
    tcav_cfg = replace(tcav_cfg, split_name=str(cfg["split_name"]))

    system_id = str(cfg["system_id"])
    layer_name = str(cfg["layer_name"])
    out_dir = REPO_ROOT / "resnet_293" / "ace" / "ace_a11_poc" / "output" / str(cfg["output_subdir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup-v3] system={system_id} layer={layer_name} device={device}")

    model = load_model(tcav_cfg, system_id).to(device)
    model.eval()
    manifest = load_manifest(tcav_cfg, system_id)
    tar_index = build_tar_index(tcav_cfg, system_id)
    shared = shared_speakers_for_both_classes(tcav_cfg, system_id, manifest)

    spoof_items = _collect_regions_for_class_v3(
        cfg=cfg,
        tcav_cfg=tcav_cfg,
        model=model,
        tar_index=tar_index,
        manifest=manifest,
        system_id=system_id,
        target_class="spoof",
        shared_speakers=shared,
        layer_name=layer_name,
    )
    bona_items = _collect_regions_for_class_v3(
        cfg=cfg,
        tcav_cfg=tcav_cfg,
        model=model,
        tar_index=tar_index,
        manifest=manifest,
        system_id=system_id,
        target_class="bonafide",
        shared_speakers=shared,
        layer_name=layer_name,
    )
    print(f"[regions-v3] spoof={len(spoof_items)} bonafide={len(bona_items)}")
    if len(spoof_items) == 0 or len(bona_items) == 0:
        raise RuntimeError("No regions collected for spoof or bonafide.")

    embedder = LayerEmbedder(model=model, layer_name=layer_name, device=device)
    try:
        print("[embed-v3] spoof")
        spoof_emb = _embed_regions(spoof_items, embedder=embedder, batch_size=128)
        print("[embed-v3] bonafide")
        bona_emb = _embed_regions(bona_items, embedder=embedder, batch_size=128)
    finally:
        embedder.close()

    print(f"[embed-v3] spoof_emb={spoof_emb.shape} bona_emb={bona_emb.shape}")

    labels, score_df, cl_meta = _cluster_per_scale(
        cfg=cfg,
        spoof_items=spoof_items,
        spoof_emb=spoof_emb,
        bona_emb=bona_emb,
    )
    score_df.to_csv(out_dir / "cluster_scores.csv", index=False)
    print(f"[score-v3] kept clusters={len(score_df)}")

    _save_cluster_concepts_v3(
        out_dir=out_dir,
        spoof_items=spoof_items,
        labels=labels,
        score_df=score_df,
        top_examples_per_cluster=int(cfg["top_examples_per_cluster"]),
        save_patch_previews=bool(cfg["save_patch_previews"]),
    )

    meta = {
        "config_path": str(config_path),
        "system_id": system_id,
        "layer_name": layer_name,
        "device": str(device),
        "n_spoof_regions": int(len(spoof_items)),
        "n_bonafide_regions": int(len(bona_items)),
        "embedding_dim": int(spoof_emb.shape[1]) if spoof_emb.ndim == 2 else None,
        "n_scored_clusters": int(len(score_df)),
        "cluster_meta": cl_meta,
        "output_dir": str(out_dir),
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[done-v3] output:", out_dir)
    return meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACE v3 (mask-aware, smoother concepts)")
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "resnet_293" / "ace" / "ace_a11_poc" / "configs" / "a11_ace_v3_config.yaml",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ace_v3(args.config)
