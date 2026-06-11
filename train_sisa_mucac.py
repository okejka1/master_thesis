"""
train_sisa_mucac.py — SISA training for MUCAC multi-label classification.

Sharding strategy: identity-based (all images of one person → one shard).
This guarantees that any forget identity is fully contained in a single shard,
which is the key property enabling efficient SISA unlearning at instance level.

Usage:
    python train_sisa_mucac.py --config configs/mucac.yaml \\
        --checkpoint-dir checkpoints/seed_0/mucac/sisa --seed 0
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

from models import build_resnet18_multilabel
from mucac_dataset import (
    MuCACDataset,
    _train_transform,
    _eval_transform,
    ensemble_evaluate_multilabel,
    identity_shard,
    LABEL_COLS,
)
from utils import set_seed


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--data-root",      default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--seed",           type=int,   default=None)
    p.add_argument("--sisa-shards",    type=int,   default=None)
    p.add_argument("--sisa-slices",    type=int,   default=None)
    return p.parse_args()


def merge(cfg, args):
    overrides = {
        "checkpoint_dir": args.checkpoint_dir,
        "data_root":      args.data_root,
        "num_epochs":     args.epochs,
        "batch_size":     args.batch_size,
        "seed":           args.seed,
        "sisa_shards":    args.sisa_shards,
        "sisa_slices":    args.sisa_slices,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ── Slice helpers ─────────────────────────────────────────────────────────────

def make_cumulative_slices(shard_indices: list[int],
                            num_slices: int) -> list[list[int]]:
    """Split shard into R cumulative slices (indices grow monotonically)."""
    arr    = np.array(shard_indices)
    chunks = np.array_split(arr, num_slices)
    cumulative, so_far = [], []
    for chunk in chunks:
        so_far = so_far + chunk.tolist()
        cumulative.append(list(so_far))
    return cumulative


def raw_slices(shard_indices: list[int], num_slices: int) -> list[list[int]]:
    arr = np.array(shard_indices)
    return [c.tolist() for c in np.array_split(arr, num_slices)]


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
        correct    += (logits > 0).float().cpu().eq(labels.cpu()).sum(dim=0)
        total      += imgs.size(0)

    return total_loss / total, float(correct.sum() / (total * len(LABEL_COLS)) * 100)


def save_shard_ckpt(model, optimizer, path, epoch, shard_id, slice_id, cfg):
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "num_labels":  len(LABEL_COLS),
        "label_cols":  LABEL_COLS,
        "dataset":     "MUCAC",
        "shard_id":    shard_id,
        "slice_id":    slice_id,
        "config":      cfg,
    }, path)


def train_shard(shard_id, shard_indices, train_ds, test_loader, cfg, device):
    num_slices       = cfg["sisa_slices"]
    num_epochs       = cfg["num_epochs"]
    epochs_per_slice = max(1, num_epochs // num_slices)
    shard_dir        = os.path.join(cfg["checkpoint_dir"],
                                    "sisa_mucac", f"shard_{shard_id}")
    os.makedirs(shard_dir, exist_ok=True)

    # Check if all slices already exist
    final_ckpt = os.path.join(shard_dir, f"slice_{num_slices - 1:02d}.pth")
    if os.path.exists(final_ckpt):
        print(f"  Shard {shard_id}: SKIP — all slices exist")
        ckpt  = torch.load(final_ckpt, map_location=device, weights_only=False)
        model = build_resnet18_multilabel(len(LABEL_COLS)).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model

    cumslices = make_cumulative_slices(shard_indices, num_slices)
    criterion = nn.BCEWithLogitsLoss()
    model     = build_resnet18_multilabel(len(LABEL_COLS)).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"]
    )

    print(f"\n  Shard {shard_id}  |  {len(shard_indices)} samples  |  "
          f"{num_slices} slices  |  {epochs_per_slice} epochs/slice")

    epoch_global = 0
    for slice_id, cum_idx in enumerate(cumslices):
        slice_ckpt = os.path.join(shard_dir, f"slice_{slice_id:02d}.pth")
        if os.path.exists(slice_ckpt):
            # Resume: load this slice and continue
            ckpt = torch.load(slice_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            epoch_global = ckpt["epoch"]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"],
                last_epoch=epoch_global - 1,
            )
            print(f"    Slice {slice_id}: SKIP (checkpoint exists)")
            continue

        loader = DataLoader(Subset(train_ds, cum_idx),
                            batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["num_workers"], pin_memory=True)

        print(f"    Slice {slice_id}  ({len(cum_idx)} samples)  "
              f"→ {epochs_per_slice} epochs")

        for _ in range(epochs_per_slice):
            epoch_global += 1
            tr_loss, tr_acc = train_one_epoch(
                model, loader, criterion, optimizer, device)
            scheduler.step()
            print(f"      ep {epoch_global:>3}  "
                  f"lr {scheduler.get_last_lr()[0]:.1e}  "
                  f"loss {tr_loss:.4f}  acc {tr_acc:.1f}%")

        save_shard_ckpt(model, optimizer, slice_ckpt,
                        epoch_global, shard_id, slice_id, cfg)
        print(f"      → {slice_ckpt}")

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = merge(load_config(args.config), args)

    set_seed(cfg["seed"])
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_shards = cfg["sisa_shards"]
    num_slices = cfg["sisa_slices"]
    sisa_dir   = os.path.join(cfg["checkpoint_dir"], "sisa_mucac")
    os.makedirs(sisa_dir, exist_ok=True)

    print(f"{'='*62}")
    print(f"  SISA Training — MUCAC")
    print(f"  Device      : {device}")
    print(f"  Shards (S)  : {num_shards}")
    print(f"  Slices (R)  : {num_slices}")
    print(f"  Epochs/shard: {cfg['num_epochs']}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  Seed        : {cfg['seed']}")
    print(f"  Out         : {sisa_dir}")
    print(f"{'='*62}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    base = os.path.join(cfg["data_root"], "mucac_dataset")
    train_ds = MuCACDataset(base + "/train.csv", base + "/train",
                            transform=_train_transform())
    test_ds  = MuCACDataset(base + "/test.csv",  base + "/test",
                            transform=_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=cfg["num_workers"],
                             pin_memory=True)

    # ── Sharding ──────────────────────────────────────────────────────────────
    shard_map_path = os.path.join(sisa_dir, "shard_assignments.json")
    if os.path.exists(shard_map_path):
        with open(shard_map_path) as f:
            shards = json.load(f)
        print(f"Loaded existing shard assignments from {shard_map_path}")
    else:
        shards = identity_shard(train_ds.df, num_shards, seed=cfg["seed"])
        with open(shard_map_path, "w") as f:
            json.dump(shards, f)
        print(f"Shard assignments saved → {shard_map_path}")

    for s, shard_idx in enumerate(shards):
        n_ids = train_ds.df.loc[shard_idx, "identity"].nunique()
        print(f"  Shard {s}: {len(shard_idx):,} samples  ({n_ids} identities)")

    # ── Train each shard ──────────────────────────────────────────────────────
    t_start      = time.time()
    shard_models = []

    for s in range(num_shards):
        set_seed(cfg["seed"] + s)
        model = train_shard(s, shards[s], train_ds, test_loader, cfg, device)
        shard_models.append(model)

    total_time = time.time() - t_start
    print(f"\n{'='*62}")
    print(f"  All {num_shards} shards done in {total_time:.0f}s")

    # ── Ensemble evaluation ───────────────────────────────────────────────────
    metrics = ensemble_evaluate_multilabel(shard_models, test_loader, device)
    print(f"\n  Ensemble test:  acc={metrics['mean_acc']:.1f}%  "
          f"f1={metrics['mean_f1']:.1f}%  bal={metrics['mean_bal_acc']:.1f}%")
    for col in LABEL_COLS:
        print(f"    {col}: f1={metrics[col+'_f1']:.1f}%")

    meta = {
        "dataset":      "MUCAC",
        "num_shards":   num_shards,
        "num_slices":   num_slices,
        "aggregation":  "soft_vote",
        "test_metrics": metrics,
        "total_time_s": total_time,
        "config":       cfg,
    }
    meta_path = os.path.join(sisa_dir, "ensemble_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  Ensemble metadata → {meta_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
