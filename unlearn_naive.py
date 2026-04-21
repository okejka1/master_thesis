"""
unlearn_naive.py — Naive (retrain-from-scratch) Machine Unlearning.

This is the gold-standard reference for all subsequent unlearning methods.
A fresh model is trained only on the RETAIN set (D_train \\ D_forget) so it
provably never sees the forget samples.

Usage
-----
# Locally:
    python unlearn_naive.py --config configs/cifar10.yaml

# On Google Colab (override dirs to Google Drive):
    !python unlearn_naive.py --config configs/cifar10.yaml \\
                             --checkpoint-dir /content/drive/MyDrive/master_thesis/checkpoints

# Override forget strategy:
    !python unlearn_naive.py --config configs/cifar10.yaml --forget-strategy class --forget-class 3

CLI overrides (all optional — default to values in the YAML config):
    --checkpoint-dir PATH       where to load/save .pth files
    --original-ckpt  PATH       explicit path to the trained model
                                (default: checkpoint_dir/resnet18_<dataset>_best.pth)
    --data-root      PATH
    --forget-strategy random|class
    --forget-fraction FLOAT     (active when strategy=random)
    --forget-class    INT       (active when strategy=class)
    --epochs         N
    --batch-size     N
    --seed           N
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

from models import build_resnet18
from utils import (STATS, evaluate, get_datasets, get_test_transform,
                   load_checkpoint, save_checkpoint, set_seed)


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Naive (retrain-from-scratch) machine unlearning."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory for loading/saving checkpoints. "
                             "Point at Google Drive on Colab.")
    parser.add_argument("--original-ckpt",  default=None,
                        help="Explicit path to the trained model checkpoint. "
                             "If omitted, looks for resnet18_<dataset>_best.pth "
                             "inside --checkpoint-dir.")
    parser.add_argument("--data-root",       default=None)
    parser.add_argument("--forget-strategy", default=None,
                        choices=["random", "class"])
    parser.add_argument("--forget-fraction", type=float, default=None)
    parser.add_argument("--forget-class",    type=int,   default=None)
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--batch-size",      type=int,   default=None)
    parser.add_argument("--seed",            type=int,   default=None)
    return parser.parse_args()


def merge(cfg: dict, args) -> dict:
    """Overlay non-None CLI args on top of YAML config values."""
    overrides = {
        "checkpoint_dir":  args.checkpoint_dir,
        "data_root":       args.data_root,
        "forget_strategy": args.forget_strategy,
        "forget_fraction": args.forget_fraction,
        "forget_class":    args.forget_class,
        "num_epochs":      args.epochs,
        "batch_size":      args.batch_size,
        "seed":            args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


# ── Forget/retain split ───────────────────────────────────────────────────────

def build_forget_retain_indices(train_dataset,
                                strategy: str,
                                forget_fraction: float,
                                forget_class: int,
                                seed: int) -> tuple[list, list]:
    """
    Partition train indices into (forget_indices, retain_indices).

    Parameters
    ----------
    strategy : str
        ``"random"`` — pick a random fraction of the whole training set.
        ``"class"``  — forget every sample belonging to ``forget_class``.

    Returns
    -------
    (forget_indices, retain_indices)
    """
    all_indices = list(range(len(train_dataset)))

    if strategy == "random":
        rng = random.Random(seed)
        forget_indices = rng.sample(all_indices,
                                    int(len(all_indices) * forget_fraction))
    elif strategy == "class":
        forget_indices = [i for i, (_, label) in enumerate(train_dataset)
                          if label == forget_class]
    else:
        raise ValueError(f"Unknown forget strategy: {strategy!r}")

    forget_set   = set(forget_indices)
    retain_indices = [i for i in all_indices if i not in forget_set]
    return forget_indices, retain_indices


# ── Training loop (identical to train.py's) ───────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
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
    args = parse_args()
    cfg  = merge(load_config(args.config), args)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    dataset     = cfg["dataset"]
    num_classes = 10 if dataset == "CIFAR10" else 100
    ds_tag      = dataset.lower()

    # Resolve original checkpoint path
    original_ckpt = (args.original_ckpt
                     or os.path.join(cfg["checkpoint_dir"],
                                     f"resnet18_{ds_tag}_best.pth"))

    print(f"{'='*65}")
    print(f"  Naive Unlearning — {dataset}")
    print(f"  Device          : {device}")
    print(f"  Forget strategy : {cfg['forget_strategy']}", end="")
    if cfg["forget_strategy"] == "random":
        print(f"  (fraction={cfg['forget_fraction']})")
    else:
        print(f"  (class={cfg['forget_class']})")
    print(f"  Original model  : {original_ckpt}")
    print(f"  Checkpoints     : {cfg['checkpoint_dir']}")
    print(f"{'='*65}\n")

    # ── Data ───────────────────────────────────────────────────────────────────
    # Two versions of train set:
    #   full_train   — augmented transforms   → used as base for Subset loaders
    #   full_eval    — no augmentation        → used for retain *evaluation*
    full_train, test_ds = get_datasets(dataset, cfg["data_root"])

    import torchvision
    DatasetClass = (torchvision.datasets.CIFAR10 if dataset == "CIFAR10"
                    else torchvision.datasets.CIFAR100)
    full_eval = DatasetClass(root=cfg["data_root"], train=True,
                             download=True,
                             transform=get_test_transform(dataset))

    forget_indices, retain_indices = build_forget_retain_indices(
        full_train,
        strategy=cfg["forget_strategy"],
        forget_fraction=cfg["forget_fraction"],
        forget_class=cfg["forget_class"],
        seed=cfg["seed"],
    )

    print(f"Forget set size : {len(forget_indices):,}")
    print(f"Retain set size : {len(retain_indices):,}")
    print(f"Test   set size : {len(test_ds):,}\n")

    # Loaders
    bs = cfg["batch_size"]
    retain_loader      = DataLoader(Subset(full_train, retain_indices),
                                    batch_size=bs, shuffle=True,
                                    num_workers=2, pin_memory=True)
    retain_eval_loader = DataLoader(Subset(full_eval, retain_indices),
                                    batch_size=bs, shuffle=False,
                                    num_workers=2, pin_memory=True)
    forget_loader      = DataLoader(Subset(full_eval, forget_indices),
                                    batch_size=bs, shuffle=False,
                                    num_workers=2, pin_memory=True)
    test_loader        = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                    num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    # ── Baseline: evaluate original model ──────────────────────────────────────
    print("Loading original model...")
    original_model = load_checkpoint(original_ckpt, device)

    print("\nEvaluating original model on all splits...")
    orig_retain_loss, orig_retain_acc = evaluate(original_model, retain_eval_loader, criterion, device)
    orig_forget_loss, orig_forget_acc = evaluate(original_model, forget_loader,      criterion, device)
    orig_test_loss,   orig_test_acc   = evaluate(original_model, test_loader,        criterion, device)

    print(f"\n{'Split':<15} {'Loss':>8}  {'Accuracy':>9}")
    print("-" * 36)
    print(f"{'Retain':<15} {orig_retain_loss:>8.4f}  {orig_retain_acc:>8.2f}%")
    print(f"{'Forget':<15} {orig_forget_loss:>8.4f}  {orig_forget_acc:>8.2f}%")
    print(f"{'Test':<15} {orig_test_loss:>8.4f}  {orig_test_acc:>8.2f}%")

    # ── Naive unlearning: retrain from scratch on retain set ───────────────────
    print(f"\nRetraining fresh model on retain set "
          f"({len(retain_indices):,} samples) for {cfg['num_epochs']} epochs...\n")

    naive_model     = build_resnet18(num_classes).to(device)
    naive_optimizer = optim.SGD(naive_model.parameters(),
                                lr=cfg["lr"], momentum=cfg["momentum"],
                                weight_decay=cfg["weight_decay"], nesterov=True)
    naive_scheduler = optim.lr_scheduler.MultiStepLR(naive_optimizer,
                                                     milestones=cfg["lr_milestones"],
                                                     gamma=cfg["lr_gamma"])

    naive_best_ckpt = os.path.join(
        cfg["checkpoint_dir"],
        f"resnet18_{ds_tag}_naive_unlearn_best.pth"
    )

    naive_history = {"train_loss": [], "train_acc": [],
                     "test_loss":  [], "test_acc":  []}
    best_naive_acc = 0.0

    print(f"{'Epoch':>6}  {'LR':>8}  "
          f"{'Retain Loss':>11}  {'Retain Acc':>10}  "
          f"{'Test Loss':>9}  {'Test Acc':>8}  {'Time':>6}")
    print("-" * 78)

    unlearn_start = time.time()
    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(naive_model, retain_loader,
                                          criterion, naive_optimizer, device)
        te_loss, te_acc = evaluate(naive_model, test_loader, criterion, device)
        naive_scheduler.step()

        naive_history["train_loss"].append(tr_loss)
        naive_history["train_acc"].append(tr_acc)
        naive_history["test_loss"].append(te_loss)
        naive_history["test_acc"].append(te_acc)

        current_lr = naive_scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(f"{epoch:>6}  {current_lr:>8.5f}  "
              f"{tr_loss:>11.4f}  {tr_acc:>9.2f}%  "
              f"{te_loss:>9.4f}  {te_acc:>7.2f}%  {elapsed:>5.1f}s")

        unlearn_time = time.time() - unlearn_start

        if te_acc > best_naive_acc:
            best_naive_acc = te_acc
            save_checkpoint(
                naive_model, naive_best_ckpt,
                epoch=epoch, test_acc=te_acc,
                dataset=dataset, num_classes=num_classes,
                extra={
                    "unlearning_method": "naive_retrain",
                    "forget_strategy":   cfg["forget_strategy"],
                    "forget_size":       len(forget_indices),
                    "retain_size":       len(retain_indices),
                    "history":           naive_history,
                    "config":            cfg,
                    "unlearn_time_s":    unlearn_time
                }
            )

    print(f"\nRetraining complete. Best test accuracy: {best_naive_acc:.2f}%")

    # ── Evaluate naive model ───────────────────────────────────────────────────
    print("\nLoading best naive unlearned model...")
    naive_model = load_checkpoint(naive_best_ckpt, device)

    print("\nEvaluating on all splits...")
    naive_retain_loss, naive_retain_acc = evaluate(naive_model, retain_eval_loader, criterion, device)
    naive_forget_loss, naive_forget_acc = evaluate(naive_model, forget_loader,      criterion, device)
    naive_test_loss,   naive_test_acc   = evaluate(naive_model, test_loader,        criterion, device)

    # ── Side-by-side comparison ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"{'Metric':<20} {'Original':>12}  {'Naive Retrain':>14}  {'Δ':>6}")
    print(f"{'='*62}")

    rows = [
        ("Retain Accuracy", orig_retain_acc, naive_retain_acc),
        ("Forget Accuracy", orig_forget_acc, naive_forget_acc),
        ("Test Accuracy",   orig_test_acc,   naive_test_acc),
    ]
    for name, orig, naive in rows:
        delta = naive - orig
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<20} {orig:>11.2f}%  {naive:>13.2f}%  {sign}{delta:>5.2f}%")

    print(f"{'='*62}")
    print(f"\nBest naive checkpoint: {naive_best_ckpt}")


if __name__ == "__main__":
    main()
