#!/usr/bin/env python3
"""
01_patch_sampler.py
-------------------
Generate balanced positive / negative 3-D patches (e.g., 64³) and
write them into patches/train/ or patches/val/.

Run:
    python 01_patch_sampler.py --labels data_preproc/labels_vox.csv \
        --vol-dir data_preproc/volumes --out-dir patches/train --neg-ratio 1
"""

import argparse, pathlib, random, numpy as np, pandas as pd
from tqdm import tqdm

PATCH_SIZE = 64
POS_MARGIN = 8   # extra voxels around center to ensure nodule inside patch

def sample_patch(vol, cz, cy, cx, rng):
    """Return a 64³ patch centered (approx) on given voxel coords."""
    z0 = int(max(cz - PATCH_SIZE//2, 0))
    y0 = int(max(cy - PATCH_SIZE//2, 0))
    x0 = int(max(cx - PATCH_SIZE//2, 0))
    patch = vol[z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
    # pad if at boundary
    pad = [(0, PATCH_SIZE - patch.shape[i]) for i in range(3)]
    patch = np.pad(patch, pad, mode="constant")
    return patch

def main(args):
    vol_dir   = pathlib.Path(args.vol_dir)
    out_dir   = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels)
    records = []

    rng = np.random.default_rng(0)

    for uid_path in tqdm(sorted(vol_dir.glob("*.npy"))):
        uid  = uid_path.stem
        vol  = np.load(uid_path)
        pos  = df[df.uid == uid]

        # ----- positives -----
        for _, row in pos.iterrows():
            patch = sample_patch(vol, row.z, row.y, row.x, rng)
            pfile = out_dir / f"{uid}_{int(row.z)}_{int(row.y)}_{int(row.x)}_pos.npy"
            np.save(pfile, patch)
            records.append([pfile, 1])

        # ----- negatives -----
        n_neg = len(pos) * args.neg_ratio
        nz, ny, nx = vol.shape
        for _ in range(n_neg):
            while True:
                rz = rng.integers(PATCH_SIZE//2, nz - PATCH_SIZE//2)
                ry = rng.integers(PATCH_SIZE//2, ny - PATCH_SIZE//2)
                rx = rng.integers(PATCH_SIZE//2, nx - PATCH_SIZE//2)
                # ensure far from any positive
                if all(abs(rz - pos.z) > PATCH_SIZE for _, pos in pos.iterrows()):
                    break
            patch = sample_patch(vol, rz, ry, rx, rng)
            nfile = out_dir / f"{uid}_{rz}_{ry}_{rx}_neg.npy"
            np.save(nfile, patch)
            records.append([nfile, 0])

    pd.DataFrame(records, columns=["path", "label"]).to_csv(out_dir/"patch_map.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",   required=True)
    ap.add_argument("--vol-dir",  required=True)
    ap.add_argument("--out-dir",  required=True)
    ap.add_argument("--neg-ratio", type=int, default=1,
                    help="negatives per positive")
    args = ap.parse_args()
    main(args)