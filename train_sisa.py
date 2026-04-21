"""
train_sisa.py — SISA (Sharded, Isolated, Sliced, Aggregated) training.

Trains S independent ResNet-18 models on disjoint, stratified shards of the
training set.  Each shard is further split into R sequential slices; a
checkpoint is saved after training on each cumulative slice so that future
unlearning only needs to retrain from the affected slice onward.

Usage
-----
# Locally:
    python train_sisa.py --config configs/cifar10.yaml

# Override SISA parameters from CLI:
    python train_sisa.py --config configs/cifar10.yaml \
                         --sisa-shards 10 --sisa-slices 20

# On Google Colab (override checkpoint dir to Google Drive):
    !python train_sisa.py --config configs/cifar10.yaml \
                          --checkpoint-dir /content/drive/MyDrive/master_thesis/checkpoints

CLI overrides (all optional — default to values in the YAML config):
    --checkpoint-dir PATH
    --data-root PATH
    --epochs N           total epochs *per shard* (divided across slices)
    --lr LR
    --batch-size N
    --seed N
    --sisa-shards S
    --sisa-slices R
    --sisa-aggregation METHOD   "soft_vote" | "majority_vote"
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

from models import build_resnet18
from utils import (
    ensemble_evaluate,
    evaluate,
    get_datasets,
    save_checkpoint,
    set_seed,
    stratified_shard,
)


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SISA training for ResNet-18 on CIFAR-10 / CIFAR-100."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--checkpoint-dir",    default=None)
    parser.add_argument("--data-root",         default=None)
    parser.add_argument("--epochs",            type=int,   default=None)
    parser.add_argument("--lr",                type=float, default=None)
    parser.add_argument("--batch-size",        type=int,   default=None)
    parser.add_argument("--seed",              type=int,   default=None)
    parser.add_argument("--sisa-shards",       type=int,   default=None)
    parser.add_argument("--sisa-slices",       type=int,   default=None)
    parser.add_argument("--sisa-aggregation",  default=None,
                        choices=["soft_vote", "majority_vote"])
    return parser.parse_args()


def merge(cfg: dict, args) -> dict:
    """Overlay non-None CLI arguments on top of YAML config values."""
    overrides = {
        "checkpoint_dir":    args.checkpoint_dir,
        "data_root":         args.data_root,
        "num_epochs":        args.epochs,
        "lr":                args.lr,
        "batch_size":        args.batch_size,
        "seed":              args.seed,
        "sisa_shards":       args.sisa_shards,
        "sisa_slices":       args.sisa_slices,
        "sisa_aggregation":  args.sisa_aggregation,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


# ── Training loop (same as train.py / unlearn_naive.py) ──────────────────────

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


# ── Shard training ────────────────────────────────────────────────────────────

def slice_shard(shard_indices: list[int],
                num_slices: int) -> list[list[int]]:
    """
    Divide shard indices into R sequential, cumulative slices.

    Returns
    -------
    list[list[int]]
        ``cumulative_slices[r]`` contains all indices from slices 0..r
        (i.e. the training set grows as we add slices).
    """
    arr = np.array(shard_indices)
    raw_slices = np.array_split(arr, num_slices)
    cumulative = []
    so_far = []
    for chunk in raw_slices:
        so_far = so_far + chunk.tolist()
        cumulative.append(list(so_far))
    return cumulative


def train_shard(shard_id: int,
                shard_indices: list[int],
                train_dataset,
                test_loader: DataLoader,
                cfg: dict,
                device: torch.device) -> nn.Module:
    """
    Train one shard model with incremental slicing and per-slice checkpoints.

    Parameters
    ----------
    shard_id : int
        Index of this shard (0-based).
    shard_indices : list[int]
        Indices into ``train_dataset`` belonging to this shard.
    train_dataset
        Full training dataset (with augmentation transforms).
    test_loader : DataLoader
        Test set loader (for periodic evaluation).
    cfg : dict
        Merged config.
    device : torch.device

    Returns
    -------
    nn.Module — the fully trained shard model.
    """
    dataset     = cfg["dataset"]
    ds_tag      = dataset.lower()
    num_classes = 10 if dataset == "CIFAR10" else 100
    num_slices  = cfg["sisa_slices"]
    num_epochs  = cfg["num_epochs"]

    # Epochs per slice: distribute total epochs across slices
    epochs_per_slice = max(1, num_epochs // num_slices)

    # Checkpoint directory for this shard
    shard_dir = os.path.join(cfg["checkpoint_dir"],
                             f"sisa_{ds_tag}", f"shard_{shard_id}")
    os.makedirs(shard_dir, exist_ok=True)

    # Cumulative slices
    cumulative_slices = slice_shard(shard_indices, num_slices)

    print(f"\n{'─'*60}")
    print(f"  Shard {shard_id}  |  {len(shard_indices)} samples  |  "
          f"{num_slices} slices  |  {epochs_per_slice} epochs/slice")
    print(f"  Checkpoints → {shard_dir}")
    print(f"{'─'*60}")

    # Fresh model for this shard
    model     = build_resnet18(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg["lr"],
                          momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"],
                          nesterov=True)

    # We compute LR milestones relative to total epochs across all slices
    total_epochs_so_far = 0

    for slice_id, cum_indices in enumerate(cumulative_slices):
        slice_loader = DataLoader(
            Subset(train_dataset, cum_indices),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        print(f"\n  Slice {slice_id}/{num_slices - 1}  "
              f"({len(cum_indices)} cumulative samples)  "
              f"→ {epochs_per_slice} epochs")

        # Create a fresh scheduler for this slice segment
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["lr_milestones"],
            gamma=cfg["lr_gamma"],
            last_epoch=total_epochs_so_far - 1 if total_epochs_so_far > 0 else -1,
        )

        for epoch in range(1, epochs_per_slice + 1):
            total_epochs_so_far += 1
            t0 = time.time()

            tr_loss, tr_acc = train_one_epoch(model, slice_loader,
                                              criterion, optimizer, device)
            scheduler.step()
            elapsed = time.time() - t0

            current_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch {total_epochs_so_far:>3}  "
                  f"LR {current_lr:.5f}  "
                  f"Loss {tr_loss:.4f}  "
                  f"Acc {tr_acc:.2f}%  "
                  f"({elapsed:.1f}s)")

        # Save checkpoint after this slice
        slice_ckpt = os.path.join(shard_dir, f"slice_{slice_id:02d}.pth")
        save_checkpoint(
            model, slice_ckpt,
            epoch=total_epochs_so_far,
            test_acc=0.0,  # will evaluate ensemble later
            dataset=dataset,
            num_classes=num_classes,
            extra={
                "shard_id":     shard_id,
                "slice_id":     slice_id,
                "num_slices":   num_slices,
                "shard_size":   len(shard_indices),
                "cum_samples":  len(cum_indices),
                "optim_state":  optimizer.state_dict(),
                "config":       cfg,
            },
        )

    # Evaluate this shard model individually on test set
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Shard {shard_id} test accuracy: {te_acc:.2f}%")

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = merge(load_config(args.config), args)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    dataset     = cfg["dataset"]
    ds_tag      = dataset.lower()
    num_classes = 10 if dataset == "CIFAR10" else 100
    num_shards  = cfg["sisa_shards"]
    num_slices  = cfg["sisa_slices"]
    aggregation = cfg["sisa_aggregation"]

    print(f"{'='*65}")
    print(f"  SISA Training — {dataset}")
    print(f"  Device        : {device}")
    print(f"  Classes       : {num_classes}")
    print(f"  Shards (S)    : {num_shards}")
    print(f"  Slices (R)    : {num_slices}")
    print(f"  Aggregation   : {aggregation}")
    print(f"  Epochs/shard  : {cfg['num_epochs']}")
    print(f"  Batch size    : {cfg['batch_size']}")
    print(f"  Checkpoints   : {cfg['checkpoint_dir']}")
    print(f"{'='*65}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, test_ds = get_datasets(dataset, cfg["data_root"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)

    # Extract targets for stratified sharding
    targets = np.array(train_ds.targets)

    print(f"Train samples : {len(train_ds):,}")
    print(f"Test  samples : {len(test_ds):,}\n")

    # ── Stratified sharding ───────────────────────────────────────────────────
    shards = stratified_shard(targets, num_shards, seed=cfg["seed"])

    # Save shard assignments for later use by unlearn_sisa.py
    sisa_dir = os.path.join(cfg["checkpoint_dir"], f"sisa_{ds_tag}")
    os.makedirs(sisa_dir, exist_ok=True)
    shard_map_path = os.path.join(sisa_dir, "shard_assignments.json")
    with open(shard_map_path, "w") as f:
        json.dump(shards, f)
    print(f"Shard assignments saved → {shard_map_path}")

    for s, shard_idx in enumerate(shards):
        class_counts = np.bincount(targets[shard_idx], minlength=num_classes)
        print(f"  Shard {s}: {len(shard_idx):,} samples  "
              f"(classes represented: {(class_counts > 0).sum()}/{num_classes})")

    # ── Train each shard ──────────────────────────────────────────────────────
    total_start = time.time()
    shard_models = []

    for s in range(num_shards):
        # Use a different seed per shard for weight init diversity
        set_seed(cfg["seed"] + s)
        model = train_shard(s, shards[s], train_ds, test_loader, cfg, device)
        shard_models.append(model)

    total_time = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  All {num_shards} shards trained in {total_time:.1f}s")
    print(f"{'='*65}")

    # ── Ensemble evaluation ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # Use NLLLoss for soft_vote since we pass log-probs
    if aggregation == "soft_vote":
        ens_criterion = nn.NLLLoss()
    else:
        ens_criterion = criterion

    ens_loss, ens_acc = ensemble_evaluate(
        shard_models, test_loader, ens_criterion, device, method=aggregation
    )

    print(f"\n{'='*65}")
    print(f"  Ensemble Test Results ({aggregation})")
    print(f"  Loss     : {ens_loss:.4f}")
    print(f"  Accuracy : {ens_acc:.2f}%")
    print(f"{'='*65}")

    # Also report individual shard accuracies
    print(f"\n  Individual shard accuracies:")
    for s, model in enumerate(shard_models):
        _, acc = evaluate(model, test_loader, criterion, device)
        print(f"    Shard {s}: {acc:.2f}%")

    # Save ensemble metadata
    meta = {
        "dataset":        dataset,
        "num_shards":     num_shards,
        "num_slices":     num_slices,
        "aggregation":    aggregation,
        "ensemble_acc":   ens_acc,
        "ensemble_loss":  ens_loss,
        "total_time_s":   total_time,
        "config":         cfg,
    }
    meta_path = os.path.join(sisa_dir, "ensemble_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nEnsemble metadata → {meta_path}")


if __name__ == "__main__":
    main()
