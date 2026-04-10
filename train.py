"""
train.py — Baseline ResNet-18 training on CIFAR-10 / CIFAR-100.

Usage
-----
# Locally:
    python train.py --config configs/cifar10.yaml

# On Google Colab (override checkpoint dir to Google Drive):
    !python train.py --config configs/cifar10.yaml \\
                     --checkpoint-dir /content/drive/MyDrive/master_thesis/checkpoints

CLI overrides (all optional — default to values in the YAML config):
    --checkpoint-dir PATH   where to save .pth files
    --data-root PATH        where to cache the dataset
    --epochs N
    --lr LR
    --batch-size N
    --seed N
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from models import build_resnet18
from utils import evaluate, get_datasets, per_class_accuracy, save_checkpoint, set_seed


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 on CIFAR-10 or CIFAR-100."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g. configs/cifar10.yaml).")
    # Optional CLI overrides — all fall back to YAML values if not specified
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Where to save checkpoints. "
                             "Useful for pointing at Google Drive on Colab.")
    parser.add_argument("--data-root",   default=None)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    return parser.parse_args()


def merge(cfg: dict, args) -> dict:
    """Overlay non-None CLI arguments on top of YAML config values."""
    overrides = {
        "checkpoint_dir": args.checkpoint_dir,
        "data_root":      args.data_root,
        "num_epochs":     args.epochs,
        "lr":             args.lr,
        "batch_size":     args.batch_size,
        "seed":           args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    """
    Run one full pass over the training loader and update model weights.

    Returns
    -------
    (avg_loss, accuracy_percent) : tuple[float, float]
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = merge(load_config(args.config), args)

    # ── Setup ──────────────────────────────────────────────────────────────────
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    dataset     = cfg["dataset"]          # "CIFAR10" | "CIFAR100"
    num_classes = 10 if dataset == "CIFAR10" else 100

    print(f"{'='*60}")
    print(f"  Training ResNet-18 on {dataset}")
    print(f"  Device      : {device}")
    print(f"  Classes     : {num_classes}")
    print(f"  Epochs      : {cfg['num_epochs']}")
    print(f"  LR          : {cfg['lr']}  (milestones {cfg['lr_milestones']}, γ={cfg['lr_gamma']})")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  Checkpoints : {cfg['checkpoint_dir']}")
    print(f"{'='*60}\n")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_ds, test_ds = get_datasets(dataset, cfg["data_root"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples : {len(train_ds):,} | "
          f"Test samples : {len(test_ds):,} | "
          f"Batches/epoch: {len(train_loader)}\n")

    # ── Model + optimiser ──────────────────────────────────────────────────────
    model     = build_resnet18(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg["lr"],
                          momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"],
                          nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg["lr_milestones"],
                                               gamma=cfg["lr_gamma"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # ── Checkpoint paths ───────────────────────────────────────────────────────
    ds_tag    = dataset.lower()
    best_ckpt = os.path.join(cfg["checkpoint_dir"],
                             f"resnet18_{ds_tag}_best.pth")
    final_ckpt = os.path.join(cfg["checkpoint_dir"],
                              f"resnet18_{ds_tag}_final.pth")

    # ── Training ───────────────────────────────────────────────────────────────
    best_acc = 0.0
    history  = {"train_loss": [], "train_acc": [],
                 "test_loss":  [], "test_acc":  []}

    print(f"{'Epoch':>6}  {'LR':>8}  "
          f"{'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Test Loss':>9}  {'Test Acc':>8}  {'Time':>6}")
    print("-" * 72)

    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(f"{epoch:>6}  {current_lr:>8.5f}  "
              f"{tr_loss:>10.4f}  {tr_acc:>8.2f}%  "
              f"{te_loss:>9.4f}  {te_acc:>7.2f}%  {elapsed:>5.1f}s")

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint(
                model, best_ckpt,
                epoch=epoch, test_acc=te_acc,
                dataset=dataset, num_classes=num_classes,
                extra={"optim_state": optimizer.state_dict(),
                       "history":     history,
                       "config":      cfg},
            )

    # Save final checkpoint regardless of accuracy
    save_checkpoint(
        model, final_ckpt,
        epoch=cfg["num_epochs"], test_acc=te_acc,
        dataset=dataset, num_classes=num_classes,
        extra={"optim_state": optimizer.state_dict(),
               "history":     history,
               "config":      cfg},
    )

    print(f"\nTraining complete.")
    print(f"  Best test accuracy : {best_acc:.2f}%")
    print(f"  Best checkpoint    : {best_ckpt}")
    print(f"  Final checkpoint   : {final_ckpt}")


if __name__ == "__main__":
    main()
