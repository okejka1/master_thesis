#!/usr/bin/env python3
"""
reeval_mia_classwise.py — Re-evaluate MIA for class-wise unlearning with
the corrected protocol: non-members = test samples of the SAME class k,
not the full test set.

Only the mia_l / mia_e fields in each result JSON (before + after) are
updated. All other metrics are left unchanged.

Supported (uses already-saved models, no retraining):
  naive      — all seeds, classes 0-2
  grad_tau   — all seeds, classes 0-2
  sisa       — seed_0 only (only seed with --save-unlearned-ckpts)
  ovr_drop   — all seeds, classes 0-2 (reconstructed from archived ensemble)

Usage
-----
    python reeval_mia_classwise.py
    python reeval_mia_classwise.py --seeds 0 --classes 0 1 2 --methods naive grad_tau
    python reeval_mia_classwise.py --dry-run   # print what would be done, no writes
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mia import run_mia_suite, run_mia_suite_ensemble
from models import build_resnet18
from ovr import OvREnsemble, load_binary
from utils import class_subset_loader, get_datasets, get_test_transform, load_checkpoint

CKPT_ROOT = "./checkpoints"
DATASET   = "cifar10"
DATASET_U = "CIFAR10"


# ── Helpers ───────────────────────────────────────────────────────────────────

def result_path(seed, method, cls):
    base = os.path.join(CKPT_ROOT, f"seed_{seed}", DATASET, "class_wise", method, f"class_{cls}")
    if method == "naive":
        return os.path.join(base, f"naive_{DATASET}_results.json")
    if method == "grad_tau":
        return os.path.join(base, f"grad_tau_{DATASET}_results.json")
    return os.path.join(base, "unlearn_results.json")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, data, dry_run=False):
    if dry_run:
        print(f"    [dry-run] would write → {path}")
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)
    print(f"    written  → {path}")


def make_loaders(seed, cls, batch_size=128):
    """forget_loader (train, class k) and mia_test_loader (test, class k)."""
    import torchvision
    train_ds, test_ds = get_datasets(DATASET_U, "./data")

    # Forget loader: training samples of class k with eval transforms
    eval_transform = get_test_transform(DATASET_U)
    DatasetClass = torchvision.datasets.CIFAR10
    full_eval = DatasetClass(root="./data", train=True, download=False,
                             transform=eval_transform)
    forget_indices = np.where(np.array(full_eval.targets) == cls)[0].tolist()
    forget_loader = DataLoader(
        torch.utils.data.Subset(full_eval, forget_indices),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    mia_test_loader = class_subset_loader(test_ds, cls, batch_size)
    return forget_loader, mia_test_loader


# ── Per-method re-evaluators ──────────────────────────────────────────────────

def reeval_naive(seed, cls, device, dry_run):
    path = result_path(seed, "naive", cls)
    if not os.path.exists(path):
        print(f"  [naive] seed={seed} class={cls} — result not found, skipping")
        return

    base_ckpt   = os.path.join(CKPT_ROOT, f"seed_{seed}", DATASET, "base",
                                f"resnet18_{DATASET}_best.pth")
    after_ckpt  = os.path.join(CKPT_ROOT, f"seed_{seed}", DATASET, "class_wise",
                                "naive", f"class_{cls}",
                                f"resnet18_{DATASET}_naive_unlearn_best.pth")
    if not os.path.exists(base_ckpt) or not os.path.exists(after_ckpt):
        print(f"  [naive] seed={seed} class={cls} — model not found, skipping")
        return

    forget_loader, mia_test_loader = make_loaders(seed, cls)
    print(f"  [naive] seed={seed} class={cls}  forget={len(forget_loader.dataset)}  "
          f"mia_test={len(mia_test_loader.dataset)}")

    before_model = load_checkpoint(base_ckpt,  device)
    after_model  = load_checkpoint(after_ckpt, device)

    before_mia = run_mia_suite(before_model, forget_loader, mia_test_loader, device, seed=seed)
    after_mia  = run_mia_suite(after_model,  forget_loader, mia_test_loader, device, seed=seed)

    record = load_json(path)
    record["before"]["mia_l"] = before_mia["mia_l"]
    record["before"]["mia_e"] = before_mia["mia_e"]
    record["after"]["mia_l"]  = after_mia["mia_l"]
    record["after"]["mia_e"]  = after_mia["mia_e"]
    write_json(path, record, dry_run)


def reeval_grad_tau(seed, cls, device, dry_run):
    path = result_path(seed, "grad_tau", cls)
    if not os.path.exists(path):
        print(f"  [grad_tau] seed={seed} class={cls} — result not found, skipping")
        return

    base_ckpt  = os.path.join(CKPT_ROOT, f"seed_{seed}", DATASET, "base",
                               f"resnet18_{DATASET}_best.pth")
    after_ckpt = os.path.join(CKPT_ROOT, f"seed_{seed}", DATASET, "class_wise",
                               "grad_tau", f"class_{cls}",
                               f"resnet18_{DATASET}_grad_tau_unlearn.pth")
    if not os.path.exists(base_ckpt) or not os.path.exists(after_ckpt):
        print(f"  [grad_tau] seed={seed} class={cls} — model not found, skipping")
        return

    forget_loader, mia_test_loader = make_loaders(seed, cls)
    print(f"  [grad_tau] seed={seed} class={cls}  forget={len(forget_loader.dataset)}  "
          f"mia_test={len(mia_test_loader.dataset)}")

    before_model = load_checkpoint(base_ckpt,  device)
    after_model  = load_checkpoint(after_ckpt, device)

    before_mia = run_mia_suite(before_model, forget_loader, mia_test_loader, device, seed=seed)
    after_mia  = run_mia_suite(after_model,  forget_loader, mia_test_loader, device, seed=seed)

    record = load_json(path)
    record["before"]["mia_l"] = before_mia["mia_l"]
    record["before"]["mia_e"] = before_mia["mia_e"]
    record["after"]["mia_l"]  = after_mia["mia_l"]
    record["after"]["mia_e"]  = after_mia["mia_e"]
    write_json(path, record, dry_run)


def _load_sisa_ensemble(seed, unlearned: bool, device):
    """Load all 5 shard models. unlearned=True → slice_09_unlearned.pth."""
    sisa_dir = os.path.join(CKPT_ROOT, f"seed_{seed}", DATASET, "sisa",
                             f"sisa_{DATASET}")
    suffix = "_unlearned" if unlearned else ""
    models = []
    for s in range(5):
        ckpt_path = os.path.join(sisa_dir, f"shard_{s}", f"slice_09{suffix}.pth")
        if not os.path.exists(ckpt_path):
            return None
        ckpt = torch.load(ckpt_path, map_location=device)
        m = build_resnet18(num_classes=10).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
    return models


def reeval_sisa(seed, cls, device, dry_run):
    path = result_path(seed, "sisa", cls)
    if not os.path.exists(path):
        print(f"  [sisa] seed={seed} class={cls} — result not found, skipping")
        return

    before_models = _load_sisa_ensemble(seed, unlearned=False, device=device)
    after_models  = _load_sisa_ensemble(seed, unlearned=True,  device=device)
    if before_models is None or after_models is None:
        print(f"  [sisa] seed={seed} class={cls} — shard models not found, skipping")
        return

    forget_loader, mia_test_loader = make_loaders(seed, cls)
    print(f"  [sisa] seed={seed} class={cls}  forget={len(forget_loader.dataset)}  "
          f"mia_test={len(mia_test_loader.dataset)}")

    before_mia = run_mia_suite_ensemble(before_models, forget_loader, mia_test_loader,
                                        device, aggregation="soft_vote", seed=seed)
    after_mia  = run_mia_suite_ensemble(after_models,  forget_loader, mia_test_loader,
                                        device, aggregation="soft_vote", seed=seed)

    record = load_json(path)
    record["before"]["mia_l"] = before_mia["mia_l"]
    record["before"]["mia_e"] = before_mia["mia_e"]
    record["after"]["mia_l"]  = after_mia["mia_l"]
    record["after"]["mia_e"]  = after_mia["mia_e"]
    write_json(path, record, dry_run)


def _load_ovr_ensemble(seed, device):
    """Load the archived (noslices) OvR ensemble."""
    ovr_dir = os.path.join(CKPT_ROOT, f"seed_{seed}", DATASET, "ovr",
                            f"ovr_{DATASET}_noslices")
    if not os.path.exists(ovr_dir):
        return None
    submodels = []
    for i in range(10):
        p = os.path.join(ovr_dir, f"class_{i}", "model.pth")
        if not os.path.exists(p):
            return None
        submodels.append(load_binary(p, device))
    return OvREnsemble(submodels, num_classes=10).to(device)


def reeval_ovr_drop(seed, cls, device, dry_run):
    path = result_path(seed, "ovr_drop", cls)
    if not os.path.exists(path):
        print(f"  [ovr_drop] seed={seed} class={cls} — result not found, skipping")
        return

    before_ensemble = _load_ovr_ensemble(seed, device)
    if before_ensemble is None:
        print(f"  [ovr_drop] seed={seed} class={cls} — ensemble not found, skipping")
        return

    # Load a fresh ensemble for "after" so drop doesn't affect "before"
    after_ensemble = _load_ovr_ensemble(seed, device)
    after_ensemble.drop(cls)

    forget_loader, mia_test_loader = make_loaders(seed, cls)
    print(f"  [ovr_drop] seed={seed} class={cls}  forget={len(forget_loader.dataset)}  "
          f"mia_test={len(mia_test_loader.dataset)}")

    before_mia = run_mia_suite(before_ensemble, forget_loader, mia_test_loader,
                               device, seed=seed)
    after_mia  = run_mia_suite(after_ensemble,  forget_loader, mia_test_loader,
                               device, seed=seed)

    record = load_json(path)
    record["before"]["mia_l"] = before_mia["mia_l"]
    record["before"]["mia_e"] = before_mia["mia_e"]
    record["after"]["mia_l"]  = after_mia["mia_l"]
    record["after"]["mia_e"]  = after_mia["mia_e"]
    write_json(path, record, dry_run)


# ── Entry point ───────────────────────────────────────────────────────────────

METHOD_FN = {
    "naive":    reeval_naive,
    "grad_tau": reeval_grad_tau,
    "sisa":     reeval_sisa,
    "ovr_drop": reeval_ovr_drop,
}

SISA_SEEDS = {0, 11, 42}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",   nargs="+", type=int,  default=[0, 11, 42])
    p.add_argument("--classes", nargs="+", type=int,  default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", type=str,
                   default=["naive", "grad_tau", "sisa", "ovr_drop"])
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without writing any files.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    for method in args.methods:
        fn = METHOD_FN.get(method)
        if fn is None:
            print(f"Unknown method {method!r}, skipping")
            continue
        print(f"\n{'='*55}\n  {method}\n{'='*55}")
        for seed in args.seeds:
            if method == "sisa" and seed not in SISA_SEEDS:
                print(f"  [sisa] seed={seed} — no unlearned checkpoints, skipping")
                continue
            for cls in args.classes:
                fn(seed, cls, device, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
