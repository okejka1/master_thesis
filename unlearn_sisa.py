"""
unlearn_sisa.py — SISA-based Machine Unlearning.

Given a trained SISA ensemble (produced by train_sisa.py), this script:
1. Identifies which shard(s) contain the data to forget
2. Finds the earliest affected slice in each affected shard
3. Reloads the checkpoint from the slice *before* the affected one
4. Retrains only the affected shard(s) from that checkpoint, excluding
   the forget samples
5. Re-evaluates the updated ensemble

Usage
-----
# Locally:
    python unlearn_sisa.py --config configs/cifar10.yaml

# Override forget strategy:
    python unlearn_sisa.py --config configs/cifar10.yaml \
                           --forget-strategy class --forget-class 3

# On Google Colab:
    !python unlearn_sisa.py --config configs/cifar10.yaml \
                            --checkpoint-dir /content/drive/MyDrive/master_thesis/checkpoints

CLI overrides (all optional):
    --checkpoint-dir PATH
    --data-root PATH
    --forget-strategy random|class
    --forget-fraction FLOAT
    --forget-class INT
    --epochs N
    --batch-size N
    --seed N
    --sisa-shards S
    --sisa-slices R
    --sisa-aggregation METHOD
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import torchvision
from torch.utils.data import DataLoader, Subset

from models import build_resnet18
from utils import (
    STATS,
    ensemble_evaluate,
    evaluate,
    get_datasets,
    get_test_transform,
    save_checkpoint,
    set_seed,
    stratified_shard,
)


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SISA-based machine unlearning."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--checkpoint-dir",    default=None)
    parser.add_argument("--data-root",         default=None)
    parser.add_argument("--forget-strategy",   default=None,
                        choices=["random", "class"])
    parser.add_argument("--forget-fraction",   type=float, default=None)
    parser.add_argument("--forget-class",      type=int,   default=None)
    parser.add_argument("--epochs",            type=int,   default=None)
    parser.add_argument("--batch-size",        type=int,   default=None)
    parser.add_argument("--seed",              type=int,   default=None)
    parser.add_argument("--sisa-shards",       type=int,   default=None)
    parser.add_argument("--sisa-slices",       type=int,   default=None)
    parser.add_argument("--sisa-aggregation",  default=None,
                        choices=["soft_vote", "majority_vote"])
    return parser.parse_args()


def merge(cfg: dict, args) -> dict:
    """Overlay non-None CLI args on top of YAML config values."""
    overrides = {
        "checkpoint_dir":    args.checkpoint_dir,
        "data_root":         args.data_root,
        "forget_strategy":   args.forget_strategy,
        "forget_fraction":   args.forget_fraction,
        "forget_class":      args.forget_class,
        "num_epochs":        args.epochs,
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


# ── Forget/retain helpers ─────────────────────────────────────────────────────

def build_forget_indices(train_dataset,
                         strategy: str,
                         forget_fraction: float,
                         forget_class: int,
                         seed: int) -> list[int]:
    """
    Compute the global forget indices (same logic as unlearn_naive.py).
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

    return forget_indices


# ── Training loop ─────────────────────────────────────────────────────────────

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


# ── Slice logic ───────────────────────────────────────────────────────────────

def slice_shard_indices(shard_indices: list[int],
                        num_slices: int) -> list[list[int]]:
    """
    Return raw (non-cumulative) slices for a shard.

    Returns
    -------
    list[list[int]]
        ``raw_slices[r]`` contains only the indices added in slice r.
    """
    arr = np.array(shard_indices)
    return [chunk.tolist() for chunk in np.array_split(arr, num_slices)]


def find_earliest_affected_slice(raw_slices: list[list[int]],
                                 forget_set: set[int]) -> int:
    """
    Find the index of the earliest slice containing any forget sample.

    Returns
    -------
    int or -1 if no slice is affected.
    """
    for r, slice_indices in enumerate(raw_slices):
        if forget_set.intersection(slice_indices):
            return r
    return -1


# ── Shard retraining ──────────────────────────────────────────────────────────

def retrain_shard(shard_id: int,
                  shard_indices: list[int],
                  forget_set: set[int],
                  train_dataset,
                  cfg: dict,
                  device: torch.device) -> tuple[nn.Module, dict]:
    """
    Retrain an affected shard from the most recent clean slice checkpoint.

    Steps:
    1. Compute raw slices for this shard
    2. Find the earliest affected slice
    3. Load the checkpoint from the slice *before* the affected one
       (or train from scratch if slice 0 is affected)
    4. Retrain on the retain data (shard - forget) from that slice onward

    Returns
    -------
    (model, stats) — the retrained model and a dictionary of stats.
    """
    dataset     = cfg["dataset"]
    ds_tag      = dataset.lower()
    num_classes = 10 if dataset == "CIFAR10" else 100
    num_slices  = cfg["sisa_slices"]
    num_epochs  = cfg["num_epochs"]
    epochs_per_slice = max(1, num_epochs // num_slices)

    shard_dir = os.path.join(cfg["checkpoint_dir"],
                             f"sisa_{ds_tag}", f"shard_{shard_id}")

    # Compute raw slices
    raw_slices = slice_shard_indices(shard_indices, num_slices)

    # Find earliest affected slice
    affected_slice = find_earliest_affected_slice(raw_slices, forget_set)
    if affected_slice == -1:
        # This shard has no forget data — should not happen if called correctly
        print(f"  Shard {shard_id}: no forget data found — skipping")
        # Load final model
        final_ckpt = os.path.join(shard_dir, f"slice_{num_slices - 1:02d}.pth")
        ckpt = torch.load(final_ckpt, map_location=device)
        model = build_resnet18(num_classes).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model, {"retrained": False}

    print(f"\n  Shard {shard_id}: earliest affected slice = {affected_slice}")

    # Load checkpoint from the slice BEFORE the affected one
    if affected_slice == 0:
        print(f"  Shard {shard_id}: slice 0 affected → retraining from scratch")
        model = build_resnet18(num_classes).to(device)
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg["lr"],
                              momentum=cfg["momentum"],
                              weight_decay=cfg["weight_decay"],
                              nesterov=True)
        start_epoch = 0
    else:
        prev_ckpt = os.path.join(shard_dir,
                                 f"slice_{affected_slice - 1:02d}.pth")
        print(f"  Shard {shard_id}: loading checkpoint → {prev_ckpt}")
        ckpt  = torch.load(prev_ckpt, map_location=device)
        model = build_resnet18(num_classes).to(device)
        model.load_state_dict(ckpt["model_state"])
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg["lr"],
                              momentum=cfg["momentum"],
                              weight_decay=cfg["weight_decay"],
                              nesterov=True)
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"]

    criterion = nn.CrossEntropyLoss()

    # Build cumulative retain indices for slices from affected_slice onward
    # (remove forget samples from every future slice)
    slices_to_retrain = list(range(affected_slice, num_slices))
    num_retrain_slices = len(slices_to_retrain)

    # For the cumulative base, gather all indices from slices *before* affected
    base_indices = []
    for r in range(affected_slice):
        retain_in_slice = [i for i in raw_slices[r] if i not in forget_set]
        base_indices.extend(retain_in_slice)

    total_retrain_epochs = 0
    t0 = time.time()

    for r in slices_to_retrain:
        # Add retain indices from slice r
        retain_in_slice = [i for i in raw_slices[r] if i not in forget_set]
        base_indices.extend(retain_in_slice)

        slice_loader = DataLoader(
            Subset(train_dataset, base_indices),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        print(f"    Retraining slice {r}/{num_slices - 1}  "
              f"({len(base_indices)} retain samples)  "
              f"→ {epochs_per_slice} epochs")

        # Create scheduler for this segment
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["lr_milestones"],
            gamma=cfg["lr_gamma"],
            last_epoch=start_epoch + total_retrain_epochs - 1
            if (start_epoch + total_retrain_epochs) > 0 else -1,
        )

        for epoch in range(1, epochs_per_slice + 1):
            total_retrain_epochs += 1
            tr_loss, tr_acc = train_one_epoch(model, slice_loader,
                                              criterion, optimizer, device)
            scheduler.step()
            print(f"      Epoch {start_epoch + total_retrain_epochs:>3}  "
                  f"Loss {tr_loss:.4f}  Acc {tr_acc:.2f}%")

        # Save updated checkpoint for this slice
        slice_ckpt = os.path.join(shard_dir,
                                  f"slice_{r:02d}_unlearned.pth")
        save_checkpoint(
            model, slice_ckpt,
            epoch=start_epoch + total_retrain_epochs,
            test_acc=0.0,
            dataset=dataset,
            num_classes=num_classes,
            extra={
                "shard_id":            shard_id,
                "slice_id":            r,
                "unlearning_method":   "sisa",
                "forget_size_shard":   len(forget_set.intersection(shard_indices)),
                "retain_size_shard":   len(base_indices),
                "optim_state":         optimizer.state_dict(),
                "config":              cfg,
            },
        )

    retrain_time = time.time() - t0

    stats = {
        "retrained":            True,
        "shard_id":             shard_id,
        "affected_slice":       affected_slice,
        "slices_retrained":     num_retrain_slices,
        "epochs_retrained":     total_retrain_epochs,
        "forget_in_shard":      len(forget_set.intersection(set(shard_indices))),
        "retrain_time_s":       retrain_time,
    }

    return model, stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = merge(load_config(args.config), args)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset     = cfg["dataset"]
    ds_tag      = dataset.lower()
    num_classes = 10 if dataset == "CIFAR10" else 100
    num_shards  = cfg["sisa_shards"]
    num_slices  = cfg["sisa_slices"]
    aggregation = cfg["sisa_aggregation"]

    sisa_dir = os.path.join(cfg["checkpoint_dir"], f"sisa_{ds_tag}")

    print(f"{'='*65}")
    print(f"  SISA Unlearning — {dataset}")
    print(f"  Device          : {device}")
    print(f"  Forget strategy : {cfg['forget_strategy']}", end="")
    if cfg["forget_strategy"] == "random":
        print(f"  (fraction={cfg['forget_fraction']})")
    else:
        print(f"  (class={cfg['forget_class']})")
    print(f"  Shards (S)      : {num_shards}")
    print(f"  Slices (R)      : {num_slices}")
    print(f"  Aggregation     : {aggregation}")
    print(f"  Checkpoints     : {sisa_dir}")
    print(f"{'='*65}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    full_train, test_ds = get_datasets(dataset, cfg["data_root"])

    DatasetClass = (torchvision.datasets.CIFAR10 if dataset == "CIFAR10"
                    else torchvision.datasets.CIFAR100)
    full_eval = DatasetClass(root=cfg["data_root"], train=True,
                             download=True,
                             transform=get_test_transform(dataset))

    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)

    # ── Load shard assignments ────────────────────────────────────────────────
    shard_map_path = os.path.join(sisa_dir, "shard_assignments.json")
    if not os.path.exists(shard_map_path):
        raise FileNotFoundError(
            f"Shard assignments not found at {shard_map_path}. "
            f"Run train_sisa.py first."
        )
    with open(shard_map_path) as f:
        shards = json.load(f)

    print(f"Loaded shard assignments from {shard_map_path}")
    print(f"  {num_shards} shards, sizes: "
          f"{[len(s) for s in shards]}\n")

    # ── Build forget set ──────────────────────────────────────────────────────
    forget_indices = build_forget_indices(
        full_train,
        strategy=cfg["forget_strategy"],
        forget_fraction=cfg["forget_fraction"],
        forget_class=cfg["forget_class"],
        seed=cfg["seed"],
    )
    forget_set = set(forget_indices)
    retain_indices = [i for i in range(len(full_train)) if i not in forget_set]

    print(f"Forget set   : {len(forget_indices):,} samples")
    print(f"Retain set   : {len(retain_indices):,} samples")
    print(f"Test set     : {len(test_ds):,} samples\n")

    # ── Load original ensemble ────────────────────────────────────────────────
    print("Loading original SISA ensemble...")
    criterion     = nn.CrossEntropyLoss()
    original_models = []

    for s in range(num_shards):
        shard_dir_s = os.path.join(sisa_dir, f"shard_{s}")
        final_ckpt  = os.path.join(shard_dir_s,
                                   f"slice_{num_slices - 1:02d}.pth")
        ckpt  = torch.load(final_ckpt, map_location=device)
        model = build_resnet18(num_classes).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        original_models.append(model)
        print(f"  Loaded shard {s} from {final_ckpt}")

    # Evaluate original ensemble
    if aggregation == "soft_vote":
        ens_criterion = nn.NLLLoss()
    else:
        ens_criterion = criterion

    # Data loaders for evaluation
    forget_loader = DataLoader(Subset(full_eval, forget_indices),
                               batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=2, pin_memory=True)
    retain_eval_loader = DataLoader(Subset(full_eval, retain_indices),
                                    batch_size=cfg["batch_size"], shuffle=False,
                                    num_workers=2, pin_memory=True)

    print("\nEvaluating original ensemble...")
    orig_test_loss,   orig_test_acc   = ensemble_evaluate(
        original_models, test_loader, ens_criterion, device, aggregation)
    orig_retain_loss, orig_retain_acc = ensemble_evaluate(
        original_models, retain_eval_loader, ens_criterion, device, aggregation)
    orig_forget_loss, orig_forget_acc = ensemble_evaluate(
        original_models, forget_loader, ens_criterion, device, aggregation)

    print(f"\n{'Split':<15} {'Loss':>8}  {'Accuracy':>9}")
    print("-" * 36)
    print(f"{'Retain':<15} {orig_retain_loss:>8.4f}  {orig_retain_acc:>8.2f}%")
    print(f"{'Forget':<15} {orig_forget_loss:>8.4f}  {orig_forget_acc:>8.2f}%")
    print(f"{'Test':<15} {orig_test_loss:>8.4f}  {orig_test_acc:>8.2f}%")

    # ── Identify affected shards ──────────────────────────────────────────────
    affected_shards = []
    for s, shard_idx in enumerate(shards):
        overlap = forget_set.intersection(shard_idx)
        if overlap:
            affected_shards.append(s)
            print(f"\n  Shard {s} is AFFECTED ({len(overlap)} forget samples)")

    unaffected_shards = [s for s in range(num_shards)
                         if s not in affected_shards]
    print(f"\n  Affected shards  : {affected_shards}")
    print(f"  Unaffected shards: {unaffected_shards}")

    # ── Retrain affected shards ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Retraining {len(affected_shards)} affected shard(s)...")
    print(f"{'='*65}")

    unlearn_start = time.time()
    updated_models = list(original_models)  # shallow copy
    all_stats = []

    for s in affected_shards:
        set_seed(cfg["seed"] + s)
        model, stats = retrain_shard(
            shard_id=s,
            shard_indices=shards[s],
            forget_set=forget_set,
            train_dataset=full_train,
            cfg=cfg,
            device=device,
        )
        updated_models[s] = model
        all_stats.append(stats)

    unlearn_time = time.time() - unlearn_start

    print(f"\n  SISA unlearning completed in {unlearn_time:.1f}s")
    for stat in all_stats:
        if stat.get("retrained"):
            print(f"    Shard {stat['shard_id']}: "
                  f"affected slice {stat['affected_slice']}, "
                  f"{stat['slices_retrained']} slices retrained, "
                  f"{stat['epochs_retrained']} epochs, "
                  f"{stat['retrain_time_s']:.1f}s")

    # ── Evaluate updated ensemble ─────────────────────────────────────────────
    print(f"\nEvaluating updated ensemble...")
    new_test_loss,   new_test_acc   = ensemble_evaluate(
        updated_models, test_loader, ens_criterion, device, aggregation)
    new_retain_loss, new_retain_acc = ensemble_evaluate(
        updated_models, retain_eval_loader, ens_criterion, device, aggregation)
    new_forget_loss, new_forget_acc = ensemble_evaluate(
        updated_models, forget_loader, ens_criterion, device, aggregation)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"{'Metric':<20} {'Before SISA':>12}  {'After SISA':>12}  {'Δ':>6}")
    print(f"{'='*68}")

    rows = [
        ("Retain Accuracy", orig_retain_acc, new_retain_acc),
        ("Forget Accuracy", orig_forget_acc, new_forget_acc),
        ("Test Accuracy",   orig_test_acc,   new_test_acc),
    ]
    for name, before, after in rows:
        delta = after - before
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<20} {before:>11.2f}%  {after:>11.2f}%  "
              f"{sign}{delta:>5.2f}%")

    # Load initial training time for metadata
    sisa_meta_path = os.path.join(sisa_dir, "ensemble_meta.json")
    sisa_train_time = 0
    if os.path.exists(sisa_meta_path):
        with open(sisa_meta_path, "r") as f:
            sisa_train_time = json.load(f).get("total_time_s", 0)

    print(f"{'='*68}")
    print(f"\n  Ensemble training time : {sisa_train_time:.1f}s")
    print(f"  Unlearning time        : {unlearn_time:.1f}s")
    print(f"  Shards retrained       : {len(affected_shards)}/{num_shards}")
    total_retrain_epochs = sum(
        s.get("epochs_retrained", 0) for s in all_stats)
    print(f"  Total retrain epochs   : {total_retrain_epochs}")

    # Save unlearning results
    results = {
        "dataset":            dataset,
        "forget_strategy":    cfg["forget_strategy"],
        "forget_size":        len(forget_indices),
        "retain_size":        len(retain_indices),
        "num_shards":         num_shards,
        "num_slices":         num_slices,
        "aggregation":        aggregation,
        "affected_shards":    affected_shards,
        "unlearn_time_s":     unlearn_time,
        "sisa_train_time_s":  sisa_train_time,
        "before": {
            "test_acc":    orig_test_acc,
            "retain_acc":  orig_retain_acc,
            "forget_acc":  orig_forget_acc,
        },
        "after": {
            "test_acc":    new_test_acc,
            "retain_acc":  new_retain_acc,
            "forget_acc":  new_forget_acc,
        },
        "shard_stats":      all_stats,
    }
    results_path = os.path.join(sisa_dir, "unlearn_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {results_path}")


if __name__ == "__main__":
    main()
