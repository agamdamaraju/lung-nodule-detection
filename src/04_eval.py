#!/usr/bin/env python3
"""
04_eval.py
----------
Compute AUC, Precision-Recall, and optionally FROC (using LUNA16 eval).
"""

import argparse, torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset import LunaPatchDataset
from torch.utils.data import DataLoader
from model   import Mini3DCNN   # adjust import path if needed

def main(args):
    ds = LunaPatchDataset(args.csv, augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Mini3DCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x,y in dl:
            preds.extend(model(x.to(device)).cpu().numpy())
            labels.extend(y.numpy())

    auc  = roc_auc_score(labels, preds)
    ap   = average_precision_score(labels, preds)
    print(f"AUC: {auc:.4f}  AP: {ap:.4f}")

    # TODO: output CSV for evaluationScript if wanted

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args)