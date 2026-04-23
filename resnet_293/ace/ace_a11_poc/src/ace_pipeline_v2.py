from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from captum.attr import LayerGradCam
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[4]
TCAV_DIR = REPO_ROOT / "resnet_293" / "tcav"
sys.path.insert(0, str(TCAV_DIR))

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
class RegionItem:
    utt_id: str
    speaker_id: str
    target_class: str
    t0: int
    t1: int
    f0: int
    f1: int
    score: float
    scale_name: str
    arr_tf: np.ndarray  # masked patch, (T,F)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_gradcam_ft(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    x_btf: torch.Tensor,
    out_size: tuple[int, int] = (80, 200),
) -> np.ndarray:
    gradcam = LayerGradCam(model, target_layer)
    x = x_btf.detach().clone().requires_grad_(True)
    at = gradcam.attribute(x, target=0)  # spoof logit
    hm = (
        F.interpolate(
            at,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )  # (F,T)
    hmin, hmax = float(hm.min()), float(hm.max())
    if hmax - hmin > 1e-8:
        hm = (hm - hmin) / (hmax - hmin)
    else:
        hm = np.zeros_like(hm, dtype=np.float32)
    return hm.astype(np.float32)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """
    Simple 4-neighbor connected components.
    mask shape: (F,T), bool
    returns list of coords arrays of shape (N,2) with (f,t).
    """
    f_max, t_max = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    comps: list[np.ndarray] = []

    for f0 in range(f_max):
        for t0 in range(t_max):
            if not mask[f0, t0] or visited[f0, t0]:
                continue
            stack = [(f0, t0)]
            visited[f0, t0] = 1
            coords: list[tuple[int, int]] = []
            while stack:
                f, t = stack.pop()
                coords.append((f, t))
                if f > 0 and mask[f - 1, t] and not visited[f - 1, t]:
                    visited[f - 1, t] = 1
                    stack.append((f - 1, t))
                if f + 1 < f_max and mask[f + 1, t] and not visited[f + 1, t]:
                    visited[f + 1, t] = 1
                    stack.append((f + 1, t))
                if t > 0 and mask[f, t - 1] and not visited[f, t - 1]:
                    visited[f, t - 1] = 1
                    stack.append((f, t - 1))
                if t + 1 < t_max and mask[f, t + 1] and not visited[f, t + 1]:
                    visited[f, t + 1] = 1
                    stack.append((f, t + 1))
            comps.append(np.asarray(coords, dtype=np.int32))
    return comps


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    at0, at1, af0, af1 = a
    bt0, bt1, bf0, bf1 = b
    it0, it1 = max(at0, bt0), min(at1, bt1)
    if0, if1 = max(af0, bf0), min(af1, bf1)
    if it1 <= it0 or if1 <= if0:
        return 0.0
    inter = (it1 - it0) * (if1 - if0)
    ua = (at1 - at0) * (af1 - af0)
    ub = (bt1 - bt0) * (bf1 - bf0)
    return float(inter / max(1, (ua + ub - inter)))


def _expand_to_min(
    t0: int,
    t1: int,
    f0: int,
    f1: int,
    min_t: int,
    min_f: int,
    max_t: int,
    max_f: int,
) -> tuple[int, int, int, int]:
    tlen = t1 - t0
    flen = f1 - f0
    if tlen < min_t:
        need = min_t - tlen
        left = need // 2
        right = need - left
        t0 = max(0, t0 - left)
        t1 = min(max_t, t1 + right)
        if t1 - t0 < min_t:
            if t0 == 0:
                t1 = min(max_t, min_t)
            elif t1 == max_t:
                t0 = max(0, max_t - min_t)
    if flen < min_f:
        need = min_f - flen
        down = need // 2
        up = need - down
        f0 = max(0, f0 - down)
        f1 = min(max_f, f1 + up)
        if f1 - f0 < min_f:
            if f0 == 0:
                f1 = min(max_f, min_f)
            elif f1 == max_f:
                f0 = max(0, max_f - min_f)
    return t0, t1, f0, f1


def _zscore_patch(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def _scale_name(
    t0: int,
    t1: int,
    f0: int,
    f1: int,
    scales: list[dict[str, Any]],
    full_t: int,
    full_f: int,
) -> str:
    area_ratio = float(((t1 - t0) * (f1 - f0)) / max(1, full_t * full_f))
    for s in scales:
        lo = float(s["min_ratio"])
        hi = float(s["max_ratio"])
        if lo <= area_ratio < hi:
            return str(s["name"])
    return str(scales[-1]["name"])


def _extract_regions_from_one_utt(
    cfg: dict[str, Any],
    fbank_tf: np.ndarray,  # (T,F)
    hm_ft: np.ndarray,  # (F,T)
) -> list[tuple[int, int, int, int, float, np.ndarray]]:
    """
    Returns list of region tuples:
      (t0,t1,f0,f1,score,masked_patch_tf)
    """
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

    full_t, full_f = fbank_tf.shape
    candidates: list[tuple[int, int, int, int, float, np.ndarray]] = []

    for q in q_list:
        thr = float(np.quantile(hm_ft, q))
        mask = hm_ft >= thr
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
            candidates.append((t0, t1, f0, f1, score, patch_tf))

    # Deduplicate by IoU, keep highest-score first
    candidates.sort(key=lambda x: x[4], reverse=True)
    deduped: list[tuple[int, int, int, int, float, np.ndarray]] = []
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


def _pad_patch_to_full_mels(patch_tf: np.ndarray, f0: int, f1: int, full_f: int = 80) -> np.ndarray:
    t = patch_tf.shape[0]
    out = np.zeros((t, full_f), dtype=np.float32)
    out[:, f0:f1] = patch_tf
    return out


class LayerEmbedder:
    def __init__(self, model: torch.nn.Module, layer_name: str, device: torch.device) -> None:
        self.model = model
        self.layer_name = layer_name
        self.device = device
        self.layer_out: torch.Tensor | None = None
        layer = getattr(self.model, self.layer_name)

        def hook(_module: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            self.layer_out = out

        self.handle = layer.register_forward_hook(hook)

    def close(self) -> None:
        self.handle.remove()

    @torch.no_grad()
    def embed_batch(self, batch_tf: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(batch_tf).float().to(self.device)
        _ = self.model(x)
        if self.layer_out is None:
            raise RuntimeError("Layer hook output is None.")
        feat = self.layer_out  # (B,C,H,W)
        emb = feat.mean(dim=(2, 3))
        return emb.detach().cpu().numpy().astype(np.float32)


def _collect_regions_for_class(
    cfg: dict[str, Any],
    tcav_cfg: Any,
    model: torch.nn.Module,
    tar_index: dict[str, tuple[Path, str]],
    manifest: pd.DataFrame,
    system_id: str,
    target_class: str,
    shared_speakers: list[str],
    layer_name: str,
) -> list[RegionItem]:
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
    full_t, full_f = 200, 80
    max_total = int(cfg["max_total_regions_per_class"])
    scales = list(cfg["area_scales"])
    out: list[RegionItem] = []

    print(f"[collect-v2:{target_class}] speakers={len(subset.selected_speakers)} rows={len(rows)}")
    for i, row in rows.iterrows():
        utt_id = str(row["utt_id"])
        speaker_id = str(row["speaker_id"])
        x = load_input(work_cfg, model, [utt_id], tar_index)  # (1,T,F)
        fbank_tf = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        hm_ft = _compute_gradcam_ft(model=model, target_layer=target_layer, x_btf=x, out_size=(80, 200))
        regs = _extract_regions_from_one_utt(cfg, fbank_tf=fbank_tf, hm_ft=hm_ft)

        for t0, t1, f0, f1, sc, patch_tf in regs:
            out.append(
                RegionItem(
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
                )
            )
        if (i + 1) % 8 == 0 or i == len(rows) - 1:
            print(f"  - {i + 1}/{len(rows)} utts, regions={len(out)}")
        if len(out) >= max_total:
            out = out[:max_total]
            print(f"[collect-v2:{target_class}] reached max_total_regions_per_class={max_total}")
            break
    return out


def _embed_regions(items: list[RegionItem], embedder: LayerEmbedder, batch_size: int = 128) -> np.ndarray:
    all_emb: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    for i, it in enumerate(items):
        full_tf = _pad_patch_to_full_mels(it.arr_tf, it.f0, it.f1, full_f=80)
        batch.append(full_tf)
        if len(batch) == batch_size or i == len(items) - 1:
            max_t = max(x.shape[0] for x in batch)
            arr = np.zeros((len(batch), max_t, 80), dtype=np.float32)
            for bi, x in enumerate(batch):
                arr[bi, : x.shape[0], :] = x
            all_emb.append(embedder.embed_batch(arr))
            batch = []
            if (i + 1) % (batch_size * 5) == 0 or i == len(items) - 1:
                print(f"  - embedded {i + 1}/{len(items)}")
    return np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, 1), dtype=np.float32)


def _choose_k_auto(x_scaled: np.ndarray, k_candidates: list[int], seed: int) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    n = len(x_scaled)
    for k in k_candidates:
        if k < 2 or k >= n:
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(x_scaled)
        if len(np.unique(labels)) < 2:
            continue
        if n > 6000:
            idx = np.random.default_rng(seed).choice(n, size=6000, replace=False)
            sc = silhouette_score(x_scaled[idx], labels[idx], metric="euclidean")
        else:
            sc = silhouette_score(x_scaled, labels, metric="euclidean")
        scores[int(k)] = float(sc)
        print(f"    [auto-k] k={k} silhouette={sc:.5f}")
    if not scores:
        # conservative fallback
        k = max(2, min(8, n - 1))
        return int(k), {int(k): float("nan")}
    best = max(scores, key=scores.get)
    return int(best), scores


def _score_one_cluster(spoof_emb: np.ndarray, bona_emb: np.ndarray, center: np.ndarray) -> dict[str, float]:
    sim_sp = cosine_similarity(spoof_emb, center).reshape(-1)
    sim_bo = cosine_similarity(bona_emb, center).reshape(-1)
    y = np.concatenate([np.ones_like(sim_sp), np.zeros_like(sim_bo)])
    p = np.concatenate([sim_sp, sim_bo])
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return {
        "score_auc_spoof_vs_bona": float(auc),
        "sim_spoof_mean": float(np.mean(sim_sp)),
        "sim_bona_mean": float(np.mean(sim_bo)),
    }


def _cluster_per_scale(
    cfg: dict[str, Any],
    spoof_items: list[RegionItem],
    spoof_emb: np.ndarray,
    bona_emb: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    k_candidates = [int(k) for k in cfg["k_candidates"]]
    min_cluster_size = int(cfg["min_cluster_size"])

    labels_global = np.full((len(spoof_items),), fill_value=-1, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"scale_k_scores": {}}
    next_cluster_id = 0

    scales = sorted({it.scale_name for it in spoof_items})
    for s in scales:
        idx = np.asarray([i for i, it in enumerate(spoof_items) if it.scale_name == s], dtype=np.int32)
        if len(idx) < 20:
            print(f"[cluster:{s}] skipped (n={len(idx)})")
            continue
        x = spoof_emb[idx]
        scaler = StandardScaler()
        x_sc = scaler.fit_transform(x)
        print(f"[cluster:{s}] n={len(idx)}")
        best_k, k_scores = _choose_k_auto(x_scaled=x_sc, k_candidates=k_candidates, seed=int(cfg["seed"]))
        meta["scale_k_scores"][s] = {str(k): float(v) for k, v in k_scores.items()}
        print(f"  [cluster:{s}] selected k={best_k}")
        km = KMeans(n_clusters=best_k, random_state=int(cfg["seed"]), n_init=20)
        lab_local = km.fit_predict(x_sc)

        for c in sorted(np.unique(lab_local).tolist()):
            sub = idx[np.where(lab_local == c)[0]]
            if len(sub) < min_cluster_size:
                continue
            center = spoof_emb[sub].mean(axis=0, keepdims=True)
            sc = _score_one_cluster(spoof_emb=spoof_emb, bona_emb=bona_emb, center=center)
            cid = next_cluster_id
            next_cluster_id += 1
            labels_global[sub] = cid
            rows.append(
                {
                    "cluster_id": int(cid),
                    "scale_name": s,
                    "cluster_size": int(len(sub)),
                    **sc,
                }
            )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("score_auc_spoof_vs_bona", ascending=False).reset_index(drop=True)
    return labels_global, df, meta


def _save_cluster_concepts(
    out_dir: Path,
    spoof_items: list[RegionItem],
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
        keep = idx[: min(len(idx), int(top_examples_per_cluster))]

        meta_rows: list[dict[str, Any]] = []
        for j, ii in enumerate(keep):
            it = spoof_items[int(ii)]
            arr_ft = it.arr_tf.T.astype(np.float32)
            np.save(cdir / f"{j:06d}.npy", arr_ft)
            meta_rows.append(
                {
                    "file": f"{j:06d}.npy",
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
            preview_items = [spoof_items[int(ii)].arr_tf.T.astype(np.float32) for ii in keep[:16]]
            max_f = max(x.shape[0] for x in preview_items)
            max_t = max(x.shape[1] for x in preview_items)
            preview = np.zeros((len(preview_items), max_f, max_t), dtype=np.float32)
            for pi, x in enumerate(preview_items):
                preview[pi, : x.shape[0], : x.shape[1]] = x
            np.save(cdir / "_preview_stack.npy", preview)

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


def run_ace_v2(config_path: Path) -> dict[str, Any]:
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
    print(f"[setup-v2] system={system_id} layer={layer_name} device={device}")

    model = load_model(tcav_cfg, system_id).to(device)
    model.eval()
    manifest = load_manifest(tcav_cfg, system_id)
    tar_index = build_tar_index(tcav_cfg, system_id)
    shared = shared_speakers_for_both_classes(tcav_cfg, system_id, manifest)

    spoof_items = _collect_regions_for_class(
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
    bona_items = _collect_regions_for_class(
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
    print(f"[regions-v2] spoof={len(spoof_items)} bonafide={len(bona_items)}")

    if len(spoof_items) == 0 or len(bona_items) == 0:
        raise RuntimeError("No regions collected for spoof or bonafide.")

    embedder = LayerEmbedder(model=model, layer_name=layer_name, device=device)
    try:
        print("[embed-v2] spoof")
        spoof_emb = _embed_regions(spoof_items, embedder=embedder, batch_size=128)
        print("[embed-v2] bonafide")
        bona_emb = _embed_regions(bona_items, embedder=embedder, batch_size=128)
    finally:
        embedder.close()

    print(f"[embed-v2] spoof_emb={spoof_emb.shape} bona_emb={bona_emb.shape}")

    labels, score_df, cl_meta = _cluster_per_scale(
        cfg=cfg,
        spoof_items=spoof_items,
        spoof_emb=spoof_emb,
        bona_emb=bona_emb,
    )
    score_df.to_csv(out_dir / "cluster_scores.csv", index=False)
    print(f"[score-v2] kept clusters={len(score_df)}")

    _save_cluster_concepts(
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

    print("[done-v2] output:", out_dir)
    return meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACE v2 (Grad-CAM-guided regions)")
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "resnet_293" / "ace" / "ace_a11_poc" / "configs" / "a11_ace_v2_config.yaml",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ace_v2(args.config)
