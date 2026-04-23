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
import yaml
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
class PatchItem:
    utt_id: str
    speaker_id: str
    target_class: str
    t0: int
    t1: int
    f0: int
    f1: int
    arr_tf: np.ndarray  # (T,F)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _patches_from_fbank(
    fbank_tf: np.ndarray,
    patch_sizes: list[tuple[int, int]],
    stride_t: int,
    stride_f: int,
) -> list[tuple[int, int, int, int, np.ndarray]]:
    """
    Extract patches from fbank in (T,F).
    Returns (t0,t1,f0,f1,patch_tf).
    """
    t_max, f_max = fbank_tf.shape
    out: list[tuple[int, int, int, int, np.ndarray]] = []
    for pt, pf in patch_sizes:
        if pt > t_max or pf > f_max:
            continue
        for t0 in range(0, t_max - pt + 1, max(1, stride_t)):
            t1 = t0 + pt
            for f0 in range(0, f_max - pf + 1, max(1, stride_f)):
                f1 = f0 + pf
                patch_tf = fbank_tf[t0:t1, f0:f1]
                out.append((t0, t1, f0, f1, patch_tf))
    return out


def _pad_patch_to_full_mels(patch_tf: np.ndarray, f0: int, f1: int, full_f: int = 80) -> np.ndarray:
    """
    Pad a patch (T, pf) back into (T, full_f) so model input F stays 80.
    """
    t, pf = patch_tf.shape
    arr = np.zeros((t, full_f), dtype=np.float32)
    arr[:, f0:f1] = patch_tf.astype(np.float32)
    return arr


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
        """
        batch_tf: (B,T,F)
        returns: (B,C) pooled embedding from layer output
        """
        x = torch.from_numpy(batch_tf).float().to(self.device)
        _ = self.model(x)  # run forward to trigger hook
        if self.layer_out is None:
            raise RuntimeError(f"Hook output for layer={self.layer_name} is None.")
        feat = self.layer_out
        # feat: (B,C,H,W)
        emb = feat.mean(dim=(2, 3))  # (B,C)
        return emb.detach().cpu().numpy().astype(np.float32)


def _collect_class_patches(
    cfg: dict[str, Any],
    tcav_cfg: Any,
    model: torch.nn.Module,
    tar_index: dict[str, tuple[Path, str]],
    manifest: pd.DataFrame,
    system_id: str,
    target_class: str,
    shared_speakers: list[str],
    rng: np.random.Generator,
) -> list[PatchItem]:
    work_cfg = tcav_cfg
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

    patch_sizes = [(int(t), int(f)) for t, f in cfg["patch_sizes"]]
    stride_t = int(cfg["patch_stride_t"])
    stride_f = int(cfg["patch_stride_f"])
    max_per_utt = int(cfg["max_patches_per_utt"])
    max_total = int(cfg["max_total_patches_per_class"])

    out: list[PatchItem] = []
    rows = subset.selected_rows.copy()
    rows["utt_id"] = rows["utt_id"].astype(str)
    rows["speaker_id"] = rows["speaker_id"].astype(str)

    print(f"[collect:{target_class}] speakers={len(subset.selected_speakers)} rows={len(rows)}")

    for i, row in rows.iterrows():
        utt_id = str(row["utt_id"])
        speaker_id = str(row["speaker_id"])

        x = load_input(work_cfg, model, [utt_id], tar_index)  # (1,T,F)
        fbank_tf = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        patches = _patches_from_fbank(
            fbank_tf=fbank_tf,
            patch_sizes=patch_sizes,
            stride_t=stride_t,
            stride_f=stride_f,
        )
        if len(patches) > max_per_utt:
            idx = rng.choice(len(patches), size=max_per_utt, replace=False)
            patches = [patches[int(j)] for j in idx]

        for t0, t1, f0, f1, p_tf in patches:
            out.append(
                PatchItem(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    target_class=target_class,
                    t0=t0,
                    t1=t1,
                    f0=f0,
                    f1=f1,
                    arr_tf=p_tf,
                )
            )

        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  - {i + 1}/{len(rows)} utts, patches={len(out)}")

        if len(out) >= max_total:
            out = out[:max_total]
            print(f"[collect:{target_class}] reached max_total_patches_per_class={max_total}")
            break

    return out


def _embed_patch_items(
    items: list[PatchItem],
    embedder: LayerEmbedder,
    batch_size: int = 128,
) -> np.ndarray:
    embeds: list[np.ndarray] = []
    full_batch: list[np.ndarray] = []
    for i, it in enumerate(items):
        full_tf = _pad_patch_to_full_mels(it.arr_tf, it.f0, it.f1, full_f=80)  # (T,80)
        full_batch.append(full_tf)
        if len(full_batch) == batch_size or i == len(items) - 1:
            arr = np.stack(full_batch, axis=0).astype(np.float32)
            emb = embedder.embed_batch(arr)
            embeds.append(emb)
            full_batch = []
            if (i + 1) % (batch_size * 5) == 0 or i == len(items) - 1:
                print(f"  - embedded {i + 1}/{len(items)} patches")
    return np.concatenate(embeds, axis=0) if embeds else np.zeros((0, 1), dtype=np.float32)


def _choose_k_auto(
    x_scaled: np.ndarray,
    k_candidates: list[int],
    seed: int,
) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for k in k_candidates:
        if k < 2 or k >= len(x_scaled):
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(x_scaled)
        # silhouette on sample if huge
        if len(x_scaled) > 8000:
            idx = np.random.default_rng(seed).choice(len(x_scaled), size=8000, replace=False)
            sc = silhouette_score(x_scaled[idx], labels[idx], metric="euclidean")
        else:
            sc = silhouette_score(x_scaled, labels, metric="euclidean")
        scores[int(k)] = float(sc)
        print(f"[auto-k] k={k} silhouette={sc:.5f}")
    if not scores:
        raise RuntimeError("No valid k candidate for clustering.")
    best_k = max(scores, key=scores.get)
    return int(best_k), scores


def _score_clusters(
    spoof_emb: np.ndarray,
    bona_emb: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == c)[0]
        if len(idx) < min_cluster_size:
            continue
        center = spoof_emb[idx].mean(axis=0, keepdims=True)  # (1,D)
        sim_sp = cosine_similarity(spoof_emb, center).reshape(-1)
        sim_bo = cosine_similarity(bona_emb, center).reshape(-1)
        y = np.concatenate([np.ones_like(sim_sp), np.zeros_like(sim_bo)])
        p = np.concatenate([sim_sp, sim_bo])
        auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
        rows.append(
            {
                "cluster_id": int(c),
                "cluster_size": int(len(idx)),
                "score_auc_spoof_vs_bona": float(auc),
                "sim_spoof_mean": float(np.mean(sim_sp)),
                "sim_bona_mean": float(np.mean(sim_bo)),
            }
        )
    df = pd.DataFrame(rows).sort_values("score_auc_spoof_vs_bona", ascending=False).reset_index(drop=True)
    return df


def _save_cluster_concepts(
    out_dir: Path,
    spoof_items: list[PatchItem],
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

        keep = idx[: min(len(idx), top_examples_per_cluster)]
        meta_rows: list[dict[str, Any]] = []
        for j, pi in enumerate(keep):
            item = spoof_items[int(pi)]
            # Save in (F,T) for compatibility with existing concept conventions
            arr_ft = item.arr_tf.T.astype(np.float32)
            np.save(cdir / f"{j:06d}.npy", arr_ft)
            meta_rows.append(
                {
                    "file": f"{j:06d}.npy",
                    "utt_id": item.utt_id,
                    "speaker_id": item.speaker_id,
                    "t0": int(item.t0),
                    "t1": int(item.t1),
                    "f0": int(item.f0),
                    "f1": int(item.f1),
                }
            )

        if save_patch_previews and len(keep) > 0:
            preview = np.stack([spoof_items[int(pi)].arr_tf.T for pi in keep[:16]], axis=0).astype(np.float32)
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


def run_ace_poc(config_path: Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    _set_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))

    tcav_cfg = load_config()
    validate_config(tcav_cfg)
    tcav_cfg.split_name = str(cfg["split_name"])

    system_id = str(cfg["system_id"])
    layer_name = str(cfg["layer_name"])

    out_dir = REPO_ROOT / "resnet_293" / "ace" / "ace_a11_poc" / "output" / str(cfg["output_subdir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] system={system_id} layer={layer_name} device={device}")

    model = load_model(tcav_cfg, system_id).to(device)
    model.eval()
    manifest = load_manifest(tcav_cfg, system_id)
    tar_index = build_tar_index(tcav_cfg, system_id)
    shared = shared_speakers_for_both_classes(tcav_cfg, system_id, manifest)

    spoof_items = _collect_class_patches(
        cfg=cfg,
        tcav_cfg=tcav_cfg,
        model=model,
        tar_index=tar_index,
        manifest=manifest,
        system_id=system_id,
        target_class="spoof",
        shared_speakers=shared,
        rng=rng,
    )
    bona_items = _collect_class_patches(
        cfg=cfg,
        tcav_cfg=tcav_cfg,
        model=model,
        tar_index=tar_index,
        manifest=manifest,
        system_id=system_id,
        target_class="bonafide",
        shared_speakers=shared,
        rng=rng,
    )

    print(f"[patches] spoof={len(spoof_items)} bonafide={len(bona_items)}")

    embedder = LayerEmbedder(model=model, layer_name=layer_name, device=device)
    try:
        print("[embed] spoof patches")
        spoof_emb = _embed_patch_items(spoof_items, embedder=embedder, batch_size=128)
        print("[embed] bonafide patches")
        bona_emb = _embed_patch_items(bona_items, embedder=embedder, batch_size=128)
    finally:
        embedder.close()

    print(f"[embed] spoof_emb={spoof_emb.shape} bona_emb={bona_emb.shape}")

    scaler = StandardScaler()
    spoof_x = scaler.fit_transform(spoof_emb)

    k_best, k_scores = _choose_k_auto(
        x_scaled=spoof_x,
        k_candidates=[int(k) for k in cfg["k_candidates"]],
        seed=int(cfg["seed"]),
    )
    print(f"[auto-k] selected k={k_best}")

    km = KMeans(n_clusters=k_best, random_state=int(cfg["seed"]), n_init=20)
    labels = km.fit_predict(spoof_x)

    score_df = _score_clusters(
        spoof_emb=spoof_emb,
        bona_emb=bona_emb,
        labels=labels,
        min_cluster_size=int(cfg["min_cluster_size"]),
    )
    score_df.to_csv(out_dir / "cluster_scores.csv", index=False)
    print(f"[score] kept clusters={len(score_df)}")

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
        "n_spoof_patches": int(len(spoof_items)),
        "n_bonafide_patches": int(len(bona_items)),
        "embedding_dim": int(spoof_emb.shape[1]) if spoof_emb.ndim == 2 else None,
        "k_candidates": [int(k) for k in cfg["k_candidates"]],
        "k_scores": {str(k): float(v) for k, v in k_scores.items()},
        "selected_k": int(k_best),
        "n_scored_clusters": int(len(score_df)),
        "output_dir": str(out_dir),
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[done] output:", out_dir)
    return meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACE A11 PoC pipeline")
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "resnet_293" / "ace" / "ace_a11_poc" / "configs" / "a11_ace_config.yaml",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ace_poc(args.config)
