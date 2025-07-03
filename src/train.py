#!/usr/bin/env python3
"""
03_train.py
-----------
(1) Standard training loop for Mini3DCNN
(2) Optuna hyper-parameter tuning with --tune N
"""

import argparse, json, os, pathlib, uuid, warnings, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler   # new API (PyTorch ≥2.3)
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import optuna
import torchio as tio
from dataset import LunaPatchDataset

warnings.filterwarnings(
    "ignore",
    message="Using TorchIO images without a torchio.SubjectsLoader",
)

# ─────────────────────────  3D CNN Skeleton ───────────────────────── #
class Mini3DCNN(nn.Module):
    def __init__(self, feat_mult: float = 1.0):
        super().__init__()
        f = lambda c: int(c * feat_mult)
        self.features = nn.Sequential(
            nn.Conv3d(1, f(32), 3, padding=1),  nn.BatchNorm3d(f(32)),  nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(f(32), f(64), 3, padding=1), nn.BatchNorm3d(f(64)), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(f(64), f(128), 3, padding=1), nn.BatchNorm3d(f(128)), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(f(128), 1)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x).squeeze(1)   # raw logits


# ────────────────────── train / eval helpers ───────────────────── #
def _epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    tot_loss, preds, labels = 0.0, [], []
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device, enabled=device == "cuda"):
            logits = model(x)
            loss   = criterion(logits, y)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        tot_loss += loss.item() * x.size(0)
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(y.cpu().numpy())

    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else np.nan
    acc = accuracy_score(labels, preds > 0.5)
    return tot_loss / len(loader.dataset), auc, acc


def train(cfg):
    train_ds = LunaPatchDataset(cfg["train_csv"], augment=True)
    val_ds = LunaPatchDataset(cfg["val_csv"],   augment=False)

    workers = min(8, os.cpu_count() or 2)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                          num_workers=workers, pin_memory=True)
    val_dl = DataLoader(val_ds,   batch_size=cfg["batch"], shuffle=False,
                          num_workers=workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Mini3DCNN(cfg["feat_mult"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    crit = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=device == "cuda")

    best_auc = 0.0
    for ep in range(cfg["epochs"]):
        tr_loss, tr_auc, _ = _epoch(model, train_dl, crit, device, optim, scaler)
        vl_loss, vl_auc, _ = _epoch(model, val_dl,   crit, device)

        print(f"[{ep+1:02d}] train {tr_loss:.4f}/{tr_auc:.3f}  "
              f"val {vl_loss:.4f}/{vl_auc:.3f}")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), f'{cfg["save_dir"]}/best.pth')
    return best_auc


# ─────────────────────────── Optuna tuning ─────────────────────────── #
def tune(args):
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(multivariate=True),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))

    def objective(trial):
        cfg = {
            "train_csv": args.train_csv,
            "val_csv":   args.val_csv,
            "save_dir":  f'{args.save_dir}/trial_{trial.number}_{uuid.uuid4().hex[:6]}',
            "epochs":    10,  # shorter per trial
            "batch":     trial.suggest_categorical("batch", [4, 8, 16]),
            "lr":        trial.suggest_loguniform("lr", 1e-5, 3e-3),
            "wd":        trial.suggest_loguniform("wd", 1e-6, 1e-2),
            "feat_mult": trial.suggest_float("feat_mult", 1.0, 2.0, step=0.5),
        }
        pathlib.Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
        auc = train(cfg)
        return auc

    study.optimize(objective, n_trials=args.tune,
                   timeout=args.timeout if args.timeout > 0 else None)
    pathlib.Path("results").mkdir(exist_ok=True)
    base = "tuning_results"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = results_dir / f"{base}_{stamp}.csv"
    study.trials_dataframe().to_csv(fname, index=False)

    print("Best params:", json.dumps(study.best_params, indent=2))
    print("Best AUC:", study.best_value)

    pathlib.Path("configs").mkdir(exist_ok=True)
    best_cfg = {
        "train_csv": args.train_csv,
        "val_csv":   args.val_csv,
        "save_dir":  f"{args.save_dir}/best_full",
        "epochs":    args.epochs,              
        **study.best_params
    }
    with open("configs/best.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
    print("Best config written to configs/best.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv",   required=True)
    parser.add_argument("--save-dir",  default="models")
    parser.add_argument("--epochs",    type=int,   default=20)
    parser.add_argument("--batch",     type=int,   default=8)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--wd",        type=float, default=1e-4)
    parser.add_argument("--feat-mult", type=float, default=1.0)
    
    parser.add_argument("--tune",    type=int, default=0,
                        help="run N Optuna trials (0 = no tuning)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Optuna time limit in seconds (0 = unlimited)")
    args = parser.parse_args()

    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.tune > 0:
        tune(args)
    else:
        cfg = vars(args) | {"train_csv": args.train_csv,
                            "val_csv":   args.val_csv,
                            "save_dir":  args.save_dir}
        best_auc = train(cfg)
        print("Final best AUC:", best_auc)