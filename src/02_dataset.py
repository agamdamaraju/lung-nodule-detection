#!/usr/bin/env python3
"""
02_dataset.py
-------------
PyTorch Dataset and transforms for patch-based CT nodule training.
"""

import numpy as np, torch
from torch.utils.data import Dataset
import pandas as pd
import torchio as tio

class LunaPatchDataset(Dataset):
    def __init__(self, csv_map, augment=False):
        self.df = pd.read_csv(csv_map)
        self.aug = tio.Compose([
            tio.RandomFlip(axes=(0,1,2)),
            tio.RandomNoise(std=(0,0.05)),
        ]) if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path  = self.df.iloc[idx,0]
        label = self.df.iloc[idx,1]
        vol   = np.load(path)  # (D,H,W)
        vol   = vol[None]      # add channel 1
        if self.aug:
            vol = self.aug(torch.tensor(vol).float())  # torchio expects Tensor
        return vol.float(), torch.tensor(label, dtype=torch.float32)

# sanity
if __name__ == "__main__":
    ds = LunaPatchDataset("../patches/train/patch_map.csv", augment=True)
    x,y = ds[0]
    print("Patch shape:", x.shape, "Label:", y)