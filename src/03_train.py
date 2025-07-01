#!/usr/bin/env python3
"""
03_train.py
-----------
Training loop for custom 3-D CNN classifier.
"""

import argparse, pathlib, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import LunaPatchDataset   # same folder import

class Mini3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1,32,3,padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64,128,3,padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(128,1)

    def forward(self,x):
        x = self.features(x)
        return torch.sigmoid(self.fc(x.flatten(1))).squeeze(1)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); running_loss = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * len(x)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval(); running_loss, preds_all, labels_all = 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss  = criterion(preds, y)
            running_loss += loss.item() * len(x)
            preds_all.append(preds.cpu()); labels_all.append(y.cpu())
    # TODO: compute AUC / accuracy
    return running_loss / len(loader.dataset)

def main(args):
    train_ds = LunaPatchDataset(args.train_csv, augment=True)
    val_ds   = LunaPatchDataset(args.val_csv,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False,num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = Mini3DCNN().to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit   = nn.BCELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_dl, optim, crit, device)
        val_loss= evaluate(model, val_dl, crit, device)
        print(f"[{epoch}] train {tr_loss:.4f}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{args.save_dir}/best.pth")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv",   required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--save-dir", default="../models")
    args = ap.parse_args()
    pathlib.Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    main(args)