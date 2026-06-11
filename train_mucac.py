"""
train_mucac.py — Baseline multi-label ResNet-18 training on MUCAC.

Usage:
    python train_mucac.py --config configs/mucac.yaml
    python train_mucac.py --config configs/mucac.yaml --seed 0 --checkpoint-dir ./checkpoints/mucac/seed_0
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from models import build_resnet18_multilabel
from mucac_dataset import (
    MuCACDataset,
    _train_transform,
    _eval_transform,
    evaluate_multilabel,
    LABEL_COLS,
)
from utils import set_seed


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--data-root",      default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--seed",           type=int,   default=None)
    return p.parse_args()


def merge(cfg: dict, args) -> dict:
    overrides = {
        "checkpoint_dir": args.checkpoint_dir,
        "data_root":      args.data_root,
        "num_epochs":     args.epochs,
        "batch_size":     args.batch_size,
        "seed":           args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = torch.zeros(len(LABEL_COLS))
    total      = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = (logits > 0).float()
        correct    += (preds == labels).sum(dim=0).cpu()
        total      += imgs.size(0)

    avg_loss = total_loss / total
    mean_acc = float(correct.sum() / (total * len(LABEL_COLS)) * 100)
    return avg_loss, mean_acc


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, path, epoch, metrics, cfg, history):
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "num_labels":  len(LABEL_COLS),
        "label_cols":  LABEL_COLS,
        "dataset":     "MUCAC",
        "test_metrics": metrics,
        "config":      cfg,
        "history":     history,
    }, path)
    print(f"  → checkpoint saved: {path}")


def load_mucac_checkpoint(path: str, device: torch.device) -> nn.Module:
    ckpt  = torch.load(path, map_location=device)
    model = build_resnet18_multilabel(ckpt["num_labels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = merge(load_config(args.config), args)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    print(f"{'='*62}")
    print(f"  Training multi-label ResNet-18 on MUCAC")
    print(f"  Device      : {device}")
    print(f"  Labels      : {LABEL_COLS}")
    print(f"  Epochs      : {cfg['num_epochs']}")
    print(f"  LR (Adam)   : {cfg['lr']}  milestones {cfg['lr_milestones']} γ={cfg['lr_gamma']}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  Seed        : {cfg['seed']}")
    print(f"  Checkpoints : {cfg['checkpoint_dir']}")
    print(f"{'='*62}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    base = os.path.join(cfg["data_root"], "mucac_dataset")
    train_ds = MuCACDataset(
        csv_path  = os.path.join(base, "train.csv"),
        img_dir   = os.path.join(base, "train"),
        transform = _train_transform(),
    )
    test_ds = MuCACDataset(
        csv_path  = os.path.join(base, "test.csv"),
        img_dir   = os.path.join(base, "test"),
        transform = _eval_transform(),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=True)

    print(f"Train: {len(train_ds):,} samples  |  Test: {len(test_ds):,} samples  "
          f"|  {len(train_loader)} batches/epoch\n")

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model     = build_resnet18_multilabel(len(LABEL_COLS)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"]
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"{'Ep':>3}  {'LR':>8}  {'TrLoss':>7}  {'TrAcc':>6}  "
          f"{'TeAcc':>6}  {'TeF1':>6}  {'TeBal':>6}  "
          f"{'Male_f1':>7}  {'Young_f1':>8}  {'Smil_f1':>7}  {'Time':>5}")
    print("-" * 90)

    history   = {"train_loss": [], "train_acc": [], "test_metrics": []}
    best_bal  = 0.0
    best_path = os.path.join(cfg["checkpoint_dir"], "mucac_best.pth")
    t_start   = time.time()

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        te_metrics = evaluate_multilabel(model, test_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_metrics"].append(te_metrics)

        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"{epoch:>3}  {lr:>8.2e}  {tr_loss:>7.4f}  {tr_acc:>5.1f}%  "
            f"{te_metrics['mean_acc']:>5.1f}%  "
            f"{te_metrics['mean_f1']:>5.1f}%  "
            f"{te_metrics['mean_bal_acc']:>5.1f}%  "
            f"{te_metrics['Male_f1']:>7.1f}%  "
            f"{te_metrics['Young_f1']:>8.1f}%  "
            f"{te_metrics['Smiling_f1']:>7.1f}%  "
            f"{elapsed:>4.0f}s"
        )

        if te_metrics["mean_bal_acc"] > best_bal:
            best_bal = te_metrics["mean_bal_acc"]
            save_checkpoint(model, best_path, epoch, te_metrics, cfg, history)

    total_time = time.time() - t_start

    final_path = os.path.join(cfg["checkpoint_dir"], "mucac_final.pth")
    save_checkpoint(model, final_path, cfg["num_epochs"], te_metrics, cfg, history)

    print(f"\nDone in {total_time:.0f}s")
    print(f"  Best mean bal_acc : {best_bal:.2f}%")
    print(f"  Best checkpoint   : {best_path}")
    print(f"  Final checkpoint  : {final_path}")


if __name__ == "__main__":
    main()
