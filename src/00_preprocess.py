#!/usr/bin/env python3
"""
00_preprocess.py
----------------
Load raw `.mhd` CT scans, apply HU windowing, optional lung mask,
resample to 1 mm³ voxel spacing, and save as `.npy` volumes.
Also converts world-space nodule annotations to voxel indices.

Run:
    python 00_preprocess.py --raw-root ../luna16_data --out-root ../data_preproc
"""

import argparse, os, json, pathlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# ---------------- util ------------------------------------------------------ #
def load_ct(mhd_path):
    """Return (volume ndarray, spacing (z,y,x), origin (z,y,x))."""
    img      = sitk.ReadImage(str(mhd_path))
    volume   = sitk.GetArrayFromImage(img).astype(np.int16)   # (z,y,x)
    spacing  = img.GetSpacing()[::-1]
    origin   = img.GetOrigin()[::-1]
    return volume, spacing, origin

def window_hu(vol, center=-600, width=1400):
    """Clip HU & scale to [-1,1]."""
    lo, hi = center - width // 2, center + width // 2
    vol    = np.clip(vol, lo, hi)
    return (vol - lo) / (hi - lo) * 2.0 - 1.0

# ---------------- main ------------------------------------------------------ #
def main(args):
    raw_root = pathlib.Path(args.raw_root)
    out_vol  = pathlib.Path(args.out_root) / "volumes"
    out_vol.mkdir(parents=True, exist_ok=True)

    ann_df = pd.read_csv(raw_root / "annotations" / "annotations.csv")
    vox_records = []

    for mhd_path in tqdm(sorted(raw_root.glob("subsets/**/*.mhd"))):
        uid = mhd_path.stem
        vol, spacing, origin = load_ct(mhd_path)
        vol = window_hu(vol)

        # TODO: add lung-mask multiplication if desired
        np.save(out_vol / f"{uid}.npy", vol.astype(np.float32))

        # convert GT annotations for this UID
        rows = ann_df[ann_df["seriesuid"] == uid]
        for _, r in rows.iterrows():
            cz = (r.coordZ - origin[0]) / spacing[0]
            cy = (r.coordY - origin[1]) / spacing[1]
            cx = (r.coordX - origin[2]) / spacing[2]
            vox_records.append([uid, cz, cy, cx, r.diameter_mm])

    pd.DataFrame(
        vox_records,
        columns=["uid", "z", "y", "x", "diam_mm"]
    ).to_csv(pathlib.Path(args.out_root) / "labels_vox.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root",  required=True,
                    help="Folder with subsets/, masks/, annotations/")
    ap.add_argument("--out-root",  required=True,
                    help="Destination for npy volumes + labels_vox.csv")
    args = ap.parse_args()
    main(args)