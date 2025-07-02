#!/usr/bin/env python3
"""
01_patch_sampler.py  (v2 – with train/val split)
------------------------------------------------
Generates balanced positive / negative 3D patches (e.g., 64³) and
writes them into patches/train/   and   patches/val/ 

Split is done 'per scan' to prevent data leakage

Example:
    python 01_patch_sampler.py \
        --labels data_preproc/labels_vox.csv \
        --vol-dir data_preproc/volumes \
        --out-root patches \
        --neg-ratio 1 \
        --val-ratio 0.2
"""

import argparse, pathlib, numpy as np, pandas as pd, random
from tqdm import tqdm

PATCH_SIZE = 64

def sample_patch(vol, cz, cy, cx, rng):
    """Extract a 64³ patch roughly centred on (cz,cy,cx)."""
    z0 = max(int(cz) - PATCH_SIZE // 2, 0)
    y0 = max(int(cy) - PATCH_SIZE // 2, 0)
    x0 = max(int(cx) - PATCH_SIZE // 2, 0)

    patch = vol[
        z0 : z0 + PATCH_SIZE,
        y0 : y0 + PATCH_SIZE,
        x0 : x0 + PATCH_SIZE,
    ]

    # pad to 64³ if near edge
    pad = [(0, PATCH_SIZE - patch.shape[i]) for i in range(3)]
    return np.pad(patch, pad, mode="constant")

def main(args):
    vol_dir  = pathlib.Path(args.vol_dir)
    out_root = pathlib.Path(args.out_root)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "val").mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    rng = np.random.default_rng(42)

    # ----------- split seriesUIDs into train / val ------------------------- #
    uids = sorted([p.stem for p in vol_dir.glob("*.npy")])
    rng.shuffle(uids)
    split_idx   = int(len(uids) * (1 - args.val_ratio))
    train_uids  = set(uids[:split_idx])
    val_uids    = set(uids[split_idx:])

    print(f"Train scans: {len(train_uids)}   Val scans: {len(val_uids)}")

    # master record lists
    rec_train, rec_val = [], []

    # ----------- iterate over every scan ----------------------------------- #
    for uid_path in tqdm(sorted(vol_dir.glob("*.npy"))):
        uid = uid_path.stem
        vol = np.load(uid_path)
        nz, ny, nx = vol.shape
        pos = labels_df[labels_df.uid == uid]

        target_dir = out_root / ("train" if uid in train_uids else "val")
        rec_collector = rec_train if uid in train_uids else rec_val

        # ----- positives ---------------------------------------------------- #
        for _, r in pos.iterrows():
            patch = sample_patch(vol, r.z, r.y, r.x, rng)
            fname = f"{uid}_{int(r.z)}_{int(r.y)}_{int(r.x)}_pos.npy"
            np.save(target_dir / fname, patch)
            rec_collector.append([str(target_dir / fname), 1])

        # ----- negatives with max-tries safeguard --------------------------- #
        n_neg = len(pos) * args.neg_ratio
        max_tries = 500
        neg_cnt = 0
        while neg_cnt < n_neg:
            tries = 0
            while tries < max_tries:
                rz = rng.integers(PATCH_SIZE // 2, nz - PATCH_SIZE // 2)
                ry = rng.integers(PATCH_SIZE // 2, ny - PATCH_SIZE // 2)
                rx = rng.integers(PATCH_SIZE // 2, nx - PATCH_SIZE // 2)

                if all(
                    abs(rz - r.z) > PATCH_SIZE
                    and abs(ry - r.y) > PATCH_SIZE
                    and abs(rx - r.x) > PATCH_SIZE
                    for _, r in pos.iterrows()
                ):
                    patch = sample_patch(vol, rz, ry, rx, rng)
                    fname = f"{uid}_{rz}_{ry}_{rx}_neg.npy"
                    np.save(target_dir / fname, patch)
                    rec_collector.append([str(target_dir / fname), 0])
                    neg_cnt += 1
                    break
                tries += 1

            if tries == max_tries:
                # print(f"[warn] skipped 1 neg for {uid} (dense positives)")
                neg_cnt += 1  

    # ----------- save CSV maps -------------------------------------------- #
    pd.DataFrame(rec_train, columns=["path", "label"]).to_csv(
        out_root / "train" / "patch_map.csv", index=False
    )
    pd.DataFrame(rec_val, columns=["path", "label"]).to_csv(
        out_root / "val" / "patch_map.csv", index=False
    )

    print("Patch sampling complete.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--vol-dir", required=True)
    parser.add_argument("--out-root", required=True,
                        help="Parent folder for patches/ (train & val subdirs)")
    parser.add_argument("--neg-ratio", type=int, default=1,
                        help="negatives per positive patch")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="fraction of scans reserved for validation")
    args = parser.parse_args()
    main(args)