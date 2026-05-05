"""Dump per-condition test/val metrics for sanity check."""
from __future__ import annotations
from pathlib import Path
import numpy as np

BASE = Path(r"Z:/users/eberrigan/SLEAP/SLEAP_Soy/2026-05-01_aug_retrain")

WANTED = ["oks_voc.mAP", "oks_voc.mAR", "vis.precision", "vis.recall", "dist.avg", "dist.p90"]


def read_metrics(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if "metrics" not in data.files:
        return {}
    m = data["metrics"].item()
    out = {}
    for k in WANTED:
        if k in m:
            try:
                out[k] = float(m[k])
            except (TypeError, ValueError):
                v = np.asarray(m[k])
                if v.dtype.kind in ("f", "i") and v.size > 0:
                    out[k] = float(np.nanmean(v))
    return out


for model_type in ("lateral", "primary"):
    print(f"\n=== {model_type} ===")
    for cond_dir in sorted((BASE / model_type).glob("*-seed0")):
        cond = cond_dir.name
        rows = []
        for split in ("train", "val", "test"):
            sf = list(cond_dir.glob(f"models/*/metrics.{split}.npz"))
            if not sf:
                continue
            metrics = read_metrics(sf[0])
            if metrics:
                rows.append((split, metrics))
        if not rows:
            print(f"  {cond}: (no metrics yet)")
            continue
        print(f"  {cond}:")
        for split, m in rows:
            line = "  ".join(f"{k.split('.')[-1]}={m[k]:.4f}" for k in WANTED if k in m)
            print(f"    {split:5s}  {line}")
