#!/usr/bin/env python3
"""
04_eval.py
----------
Evaluate a trained Mini3DCNN and, if requested, dump a LUNA16-style
candidates CSV plus ROC/PR plots.

Usage examples
--------------
# metrics only
python 04_eval.py --csv patches/val/patch_map.csv --ckpt models/best.pth --feat-mult 2.0

# metrics + candidate CSV + plots
python 04_eval.py --csv patches/val/patch_map.csv --ckpt models/best.pth \
                  --feat-mult 2.0 --out-csv results/val_candidates.csv \
                  --plot-dir plots/val
"""

import argparse, re, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
import matplotlib.pyplot as plt
import SimpleITK as sitk                         # fallback meta reader

from dataset import LunaPatchDataset
from train import Mini3DCNN                      # adjust path if needed

# UID regex (greedy UID, then z_y_x_suffix)
_UID_RE = re.compile(r'^(.*?)_(\d+)_(\d+)_(\d+)_(pos|neg)\.npy$')


@torch.inference_mode()
def main(args):
    # ───────────────────────── Dataset & Model ─────────────────────── #
    ds = LunaPatchDataset(args.csv, augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Mini3DCNN(feat_mult=args.feat_mult).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # ───────────────────────── Inference loop ──────────────────────── #
    probs, labels, paths = [], [], []
    for (x, y), p in zip(dl, ds.df.path):
        logits = model(x.to(device))
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(y.numpy())
        paths.extend(p)

    probs   = np.asarray(probs,   dtype=np.float32)
    labels  = np.asarray(labels,  dtype=np.float32)
    roc_auc = roc_auc_score(labels, probs)
    pr_auc  = average_precision_score(labels, probs)
    print(f"AUC     : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

    # ────────────────────── Optional plots ─────────────────────────── #
    if args.plot_dir:
        pdir = Path(args.plot_dir)
        pdir.mkdir(parents=True, exist_ok=True)

        # ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(); plt.plot(fpr, tpr, lw=2); plt.plot([0, 1], [0, 1], "--", lw=1)
        plt.xlabel("False-Positive Rate"); plt.ylabel("True-Positive Rate")
        plt.title(f"ROC (AUC = {roc_auc:.3f})")
        roc_png = pdir / "roc_curve.png"; plt.savefig(roc_png, bbox_inches="tight"); plt.close()

        # PR curve
        prec, rec, _ = precision_recall_curve(labels, probs)
        plt.figure(); plt.plot(rec, prec, lw=2)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall (AP = {pr_auc:.3f})")
        pr_png = pdir / "pr_curve.png"; plt.savefig(pr_png, bbox_inches="tight"); plt.close()

        print(f"✓ ROC curve saved: {roc_png}")
        print(f"✓ PR  curve saved: {pr_png}")

    # ───────────── Optional candidate CSV for LUNA evaluator ───────── #
    if args.out_csv:
        meta = {}
        vox_csv = Path("data_preproc/labels_vox.csv")
        if vox_csv.exists():
            tmp = (
                pd.read_csv(vox_csv)
                  .drop_duplicates("uid")
                  .set_index("uid")
            )
            meta = tmp.to_dict("index")   # uid -> dict with spacing_*, origin_*

        def spacing_origin(uid: str):
            """Return (spacing z,y,x) , (origin z,y,x) in mm."""
            if uid in meta and "spacing_z" in meta[uid]:
                m = meta[uid]
                sp = (m["spacing_z"], m["spacing_y"], m["spacing_x"])
                org= (m["origin_z"],  m["origin_y"],  m["origin_x"])
                return sp, org
        
            mhd = next(Path("luna16_data").rglob(f"{uid}.mhd"))
            img = sitk.ReadImage(str(mhd))
            sp  = img.GetSpacing()[::-1]
            org = img.GetOrigin()[::-1]
            return sp, org

        rows, skipped = [], 0
        for path, prob in zip(paths, probs):
            m = _UID_RE.match(Path(path).name)
            if m is None:
                skipped += 1
                continue
            uid, cz, cy, cx, _ = m.groups()
            cz, cy, cx = map(float, (cz, cy, cx))

            (sz, sy, sx), (oz, oy, ox) = spacing_origin(uid)
            wz = oz + cz * sz
            wy = oy + cy * sy
            wx = ox + cx * sx
            rows.append([uid, wx, wy, wz, prob])

        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "probability"]
                     ).to_csv(out_path, index=False)

        print(f"Candidate CSV saved: {out_path}  ({len(rows)} rows)")
        if skipped:
            print(f"[warn] skipped {skipped} filenames that didn’t match UID_Z_Y_X_*.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="patch_map.csv to evaluate")
    parser.add_argument("--ckpt", required=True, help="Mini3DCNN checkpoint (.pth)")
    parser.add_argument("--feat-mult", type=float, default=1.0,
                        help="channel multiplier used at training time")
    parser.add_argument("--plot-dir", default="",
                        help="folder to save ROC / PR PNGs (optional)")
    parser.add_argument("--out-csv",  default="",
                        help="write LUNA16 candidate CSV (optional)")
    args = parser.parse_args()
    main(args)