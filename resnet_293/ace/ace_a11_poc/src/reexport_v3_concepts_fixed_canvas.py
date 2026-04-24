from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _to_full_canvas_ft(
    patch_ft: np.ndarray,
    t0: int,
    t1: int,
    f0: int,
    f1: int,
    target_mels: int,
    target_frames: int,
) -> np.ndarray:
    """
    patch_ft: (F,T) local crop
    returns full-canvas (F,T)
    """
    full_tf = np.zeros((target_frames, target_mels), dtype=np.float32)
    patch_tf = patch_ft.T.astype(np.float32)

    tt0 = max(0, int(t0))
    tt1 = min(int(target_frames), int(t1))
    ff0 = max(0, int(f0))
    ff1 = min(int(target_mels), int(f1))
    if tt1 <= tt0 or ff1 <= ff0:
        return full_tf.T

    src_t = tt1 - tt0
    src_f = ff1 - ff0
    full_tf[tt0:tt1, ff0:ff1] = patch_tf[:src_t, :src_f]
    return full_tf.T


def reexport(
    run_dir: Path,
    target_mels: int = 80,
    target_frames: int = 200,
    overwrite: bool = False,
) -> None:
    concepts_dir = run_dir / "concepts_auto"
    if not concepts_dir.is_dir():
        raise FileNotFoundError(f"Missing concepts_auto: {concepts_dir}")

    cluster_dirs = sorted([p for p in concepts_dir.glob("cluster_*") if p.is_dir()])
    if not cluster_dirs:
        raise RuntimeError(f"No cluster dirs under {concepts_dir}")

    print(f"[reexport] clusters={len(cluster_dirs)} target=(F={target_mels},T={target_frames})")

    for cdir in cluster_dirs:
        meta_path = cdir / "meta.json"
        if not meta_path.exists():
            print(f"  - skip {cdir.name}: missing meta.json")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = meta.get("rows", [])
        for row in rows:
            fname = row.get("file")
            if not fname:
                continue
            fp = cdir / fname
            if not fp.exists():
                continue
            arr = np.load(fp).astype(np.float32)  # (F,T) local
            full = _to_full_canvas_ft(
                patch_ft=arr,
                t0=int(row.get("t0", 0)),
                t1=int(row.get("t1", 0)),
                f0=int(row.get("f0", 0)),
                f1=int(row.get("f1", 0)),
                target_mels=target_mels,
                target_frames=target_frames,
            )
            if overwrite:
                np.save(fp, full.astype(np.float32))
            else:
                np.save(cdir / f"{Path(fname).stem}_full.npy", full.astype(np.float32))

            mrel = row.get("mask_file")
            if mrel:
                mp = cdir / mrel
                if mp.exists():
                    m = np.load(mp).astype(np.float32)
                    mfull = _to_full_canvas_ft(
                        patch_ft=m,
                        t0=int(row.get("t0", 0)),
                        t1=int(row.get("t1", 0)),
                        f0=int(row.get("f0", 0)),
                        f1=int(row.get("f1", 0)),
                        target_mels=target_mels,
                        target_frames=target_frames,
                    )
                    if overwrite:
                        np.save(mp, mfull.astype(np.float32))
                    else:
                        out_mask = mp.parent / f"{mp.stem}_full.npy"
                        np.save(out_mask, mfull.astype(np.float32))

        print(f"  - done {cdir.name}")

    print("[reexport] done")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-export ACE v3 concepts to fixed full canvas.")
    p.add_argument("--run-dir", type=Path, required=True, help="ACE run dir containing concepts_auto/")
    p.add_argument("--target-mels", type=int, default=80)
    p.add_argument("--target-frames", type=int, default=200)
    p.add_argument("--overwrite", action="store_true", help="Overwrite original files instead of writing *_full.npy")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reexport(
        run_dir=args.run_dir,
        target_mels=int(args.target_mels),
        target_frames=int(args.target_frames),
        overwrite=bool(args.overwrite),
    )
