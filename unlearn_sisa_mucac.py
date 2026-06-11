"""
unlearn_sisa_mucac.py — SISA-based instance-level unlearning on MUCAC.

Usage:
    python unlearn_sisa_mucac.py --config configs/mucac.yaml \\
        --checkpoint-dir checkpoints/seed_0/mucac/sisa \\
        --output-dir     checkpoints/seed_0/mucac/instance_wise/sisa/frac_01pct \\
        --forget-fraction 0.01 --seed 0
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
    evaluate_multilabel,
    LABEL_COLS,
)
from mia import run_mia_suite_multilabel_ensemble
from unlearn_naive_mucac import build_forget_retain_indices
from utils import set_seed


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",            required=True)
    p.add_argument("--checkpoint-dir",    default=None,
                   help="Directory containing the sisa_mucac/ shard tree.")
    p.add_argument("--output-dir",        default=None,
                   help="Where to write sisa_mucac_results.json.")
    p.add_argument("--data-root",         default=None)
    p.add_argument("--forget-fraction",   type=float, default=None)
    p.add_argument("--epochs",            type=int,   default=None)
    p.add_argument("--batch-size",        type=int,   default=None)
    p.add_argument("--seed",              type=int,   default=None)
    p.add_argument("--sisa-shards",       type=int,   default=None)
    p.add_argument("--sisa-slices",       type=int,   default=None)
    p.add_argument("--save-unlearned-ckpts", action="store_true", default=False)
    return p.parse_args()


def merge(cfg, args):
    overrides = {
        "checkpoint_dir":       args.checkpoint_dir,
        "output_dir":           args.output_dir,
        "data_root":            args.data_root,
        "forget_fraction":      args.forget_fraction,
        "num_epochs":           args.epochs,
        "batch_size":           args.batch_size,
        "seed":                 args.seed,
        "sisa_shards":          args.sisa_shards,
        "sisa_slices":          args.sisa_slices,
        "save_unlearned_ckpts": args.save_unlearned_ckpts,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ── Slice helpers (same as train_sisa_mucac) ──────────────────────────────────

def raw_slices(shard_indices: list[int], num_slices: int) -> list[list[int]]:
    arr = np.array(shard_indices)
    return [c.tolist() for c in np.array_split(arr, num_slices)]


def find_earliest_affected_slice(raw_slice_list: list[list[int]],
                                  forget_set: set[int]) -> int:
    for r, chunk in enumerate(raw_slice_list):
        if forget_set.intersection(chunk):
            return r
    return -1


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total      = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)

    return total_loss / total


# ── Shard retrain ─────────────────────────────────────────────────────────────

def retrain_shard(shard_id, shard_indices, forget_set, train_ds, cfg, device):
    num_slices       = cfg["sisa_slices"]
    num_epochs       = cfg["num_epochs"]
    epochs_per_slice = max(1, num_epochs // num_slices)
    shard_dir        = os.path.join(cfg["checkpoint_dir"],
                                    "sisa_mucac", f"shard_{shard_id}")
    criterion        = nn.BCEWithLogitsLoss()

    chunks       = raw_slices(shard_indices, num_slices)
    first_bad    = find_earliest_affected_slice(chunks, forget_set)

    if first_bad == -1:
        # No forget data in this shard — load final slice as-is
        ckpt  = torch.load(os.path.join(shard_dir,
                           f"slice_{num_slices - 1:02d}.pth"),
                           map_location=device, weights_only=False)
        model = build_resnet18_multilabel(len(LABEL_COLS)).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model, {"retrained": False, "shard_id": shard_id}

    print(f"\n  Shard {shard_id}: first affected slice = {first_bad}")

    if first_bad == 0:
        model     = build_resnet18_multilabel(len(LABEL_COLS)).to(device)
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        start_ep  = 0
    else:
        prev = os.path.join(shard_dir, f"slice_{first_bad - 1:02d}.pth")
        ckpt  = torch.load(prev, map_location=device, weights_only=False)
        model = build_resnet18_multilabel(len(LABEL_COLS)).to(device)
        model.load_state_dict(ckpt["model_state"])
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_ep  = ckpt["epoch"]

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"],
        last_epoch=start_ep - 1,
    )

    base_idx   = []
    for r in range(first_bad):
        base_idx.extend(i for i in chunks[r] if i not in forget_set)

    ep_global         = start_ep
    total_retrain_eps = 0
    t0                = time.time()

    for r in range(first_bad, num_slices):
        base_idx.extend(i for i in chunks[r] if i not in forget_set)
        loader = DataLoader(Subset(train_ds, base_idx),
                            batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["num_workers"], pin_memory=True)
        print(f"    Slice {r}  ({len(base_idx)} retain samples)  "
              f"→ {epochs_per_slice} epochs")

        for _ in range(epochs_per_slice):
            ep_global         += 1
            total_retrain_eps += 1
            tr_loss = train_one_epoch(model, loader, criterion, optimizer, device)
            scheduler.step()
            print(f"      ep {ep_global:>3}  "
                  f"lr {scheduler.get_last_lr()[0]:.1e}  loss {tr_loss:.4f}")

        if cfg.get("save_unlearned_ckpts", False):
            sl_ckpt = os.path.join(shard_dir, f"slice_{r:02d}_unlearned.pth")
            torch.save({
                "epoch": ep_global, "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "num_labels": len(LABEL_COLS), "label_cols": LABEL_COLS,
            }, sl_ckpt)

    stats = {
        "retrained":        True,
        "shard_id":         shard_id,
        "first_bad_slice":  first_bad,
        "slices_retrained": num_slices - first_bad,
        "epochs_retrained": total_retrain_eps,
        "forget_in_shard":  len(forget_set.intersection(shard_indices)),
        "retrain_time_s":   time.time() - t0,
    }
    return model, stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_results(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def _flatten(test_m, retain_m, forget_m, mia):
    return ({f"test_{k}": v for k, v in test_m.items()}
            | {f"retain_{k}": v for k, v in retain_m.items()}
            | {f"forget_{k}": v for k, v in forget_m.items()}
            | {"mia_l": mia["mia_l"], "mia_e": mia["mia_e"]})


def _print_split(name, m):
    print(f"  {name:<7}  acc={m['mean_acc']:.1f}%  f1={m['mean_f1']:.1f}%  "
          f"bal={m['mean_bal_acc']:.1f}%  "
          f"[Male f1={m['Male_f1']:.1f}%  "
          f"Young f1={m['Young_f1']:.1f}%  "
          f"Smiling f1={m['Smiling_f1']:.1f}%]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = merge(load_config(args.config), args)

    frac = cfg["forget_fraction"]
    if frac is None:
        raise ValueError("--forget-fraction required")

    set_seed(cfg["seed"])
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_shards = cfg["sisa_shards"]
    num_slices = cfg["sisa_slices"]
    sisa_dir   = os.path.join(cfg["checkpoint_dir"], "sisa_mucac")

    out_dir      = cfg.get("output_dir") or sisa_dir
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "sisa_mucac_results.json")

    if os.path.exists(results_path):
        try:
            if json.load(open(results_path)).get("status") == "complete":
                print(f"[sisa-mucac] Already complete → {results_path}")
                return
        except Exception:
            pass

    print(f"{'='*62}")
    print(f"  SISA Unlearning — MUCAC")
    print(f"  Device          : {device}")
    print(f"  Forget fraction : {frac*100:.2f}%%   seed={cfg['seed']}")
    print(f"  Shards (S)      : {num_shards}   Slices (R): {num_slices}")
    print(f"  Shard tree      : {sisa_dir}")
    print(f"  Results out     : {out_dir}")
    print(f"{'='*62}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    base = os.path.join(cfg["data_root"], "mucac_dataset")
    train_aug  = MuCACDataset(base + "/train.csv", base + "/train",
                              transform=_train_transform())
    train_eval = MuCACDataset(base + "/train.csv", base + "/train",
                              transform=_eval_transform())
    test_ds    = MuCACDataset(base + "/test.csv",  base + "/test",
                              transform=_eval_transform())

    forget_idx, retain_idx, forget_ids = build_forget_retain_indices(
        train_aug.df, frac, cfg["seed"]
    )
    forget_set = set(forget_idx)

    bs, nw = cfg["batch_size"], cfg["num_workers"]
    forget_loader      = DataLoader(Subset(train_eval, forget_idx),
                                    batch_size=bs, shuffle=False,
                                    num_workers=nw, pin_memory=True)
    retain_eval_loader = DataLoader(Subset(train_eval, retain_idx),
                                    batch_size=bs, shuffle=False,
                                    num_workers=nw, pin_memory=True)
    test_loader        = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                    num_workers=nw, pin_memory=True)

    print(f"  Forget: {len(forget_idx):,} samples  ({len(forget_ids)} identities)")
    print(f"  Retain: {len(retain_idx):,} samples\n")

    # ── Load shard assignments ────────────────────────────────────────────────
    with open(os.path.join(sisa_dir, "shard_assignments.json")) as f:
        shards = json.load(f)

    # ── Load original ensemble ────────────────────────────────────────────────
    print("Loading original SISA ensemble...")
    orig_models = []
    for s in range(num_shards):
        ckpt = torch.load(
            os.path.join(sisa_dir, f"shard_{s}", f"slice_{num_slices-1:02d}.pth"),
            map_location=device, weights_only=False,
        )
        m = build_resnet18_multilabel(len(LABEL_COLS)).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        orig_models.append(m)

    print("Baseline ensemble metrics:")
    orig_test_m   = ensemble_evaluate_multilabel(orig_models, test_loader,        device)
    orig_retain_m = ensemble_evaluate_multilabel(orig_models, retain_eval_loader, device)
    orig_forget_m = ensemble_evaluate_multilabel(orig_models, forget_loader,      device)
    _print_split("Test",   orig_test_m)
    _print_split("Retain", orig_retain_m)
    _print_split("Forget", orig_forget_m)

    print("\nMIA on original ensemble:")
    orig_mia = run_mia_suite_multilabel_ensemble(
        orig_models, forget_loader, test_loader, device,
        label="Original", seed=cfg["seed"],
    )

    # ── Identify & retrain affected shards ────────────────────────────────────
    affected = [s for s, idx in enumerate(shards)
                if forget_set.intersection(idx)]
    print(f"\n  Affected shards: {affected} / {num_shards}")
    print(f"{'='*62}")
    print(f"  Retraining {len(affected)} affected shard(s)...")

    t_unlearn     = time.time()
    updated_models = list(orig_models)
    all_stats      = []

    for s in affected:
        set_seed(cfg["seed"] + s)
        model, stats = retrain_shard(
            s, shards[s], forget_set, train_aug, cfg, device
        )
        updated_models[s] = model
        all_stats.append(stats)

    unlearn_time = time.time() - t_unlearn
    print(f"\n  SISA unlearning done in {unlearn_time:.0f}s")
    for st in all_stats:
        if st.get("retrained"):
            print(f"    Shard {st['shard_id']}: "
                  f"first bad slice {st['first_bad_slice']}, "
                  f"{st['slices_retrained']} slices, "
                  f"{st['epochs_retrained']} epochs, "
                  f"{st['retrain_time_s']:.0f}s")

    # ── Evaluate updated ensemble ─────────────────────────────────────────────
    print("\nUpdated ensemble metrics:")
    new_test_m   = ensemble_evaluate_multilabel(updated_models, test_loader,        device)
    new_retain_m = ensemble_evaluate_multilabel(updated_models, retain_eval_loader, device)
    new_forget_m = ensemble_evaluate_multilabel(updated_models, forget_loader,      device)
    _print_split("Test",   new_test_m)
    _print_split("Retain", new_retain_m)
    _print_split("Forget", new_forget_m)

    print("\nMIA on updated ensemble:")
    new_mia = run_mia_suite_multilabel_ensemble(
        updated_models, forget_loader, test_loader, device,
        label="After SISA", seed=cfg["seed"],
    )

    # ── Load naive retrain time as reference ──────────────────────────────────
    meta_path = os.path.join(sisa_dir, "ensemble_meta.json")
    sisa_train_time = (json.load(open(meta_path)).get("total_time_s", 0)
                       if os.path.exists(meta_path) else 0)

    print(f"\n{'='*62}")
    print(f"  {'Metric':<24} {'Before':>9}  {'After':>9}  {'Δ':>7}")
    print(f"{'='*62}")
    for key, label in [("mean_acc", "Mean Acc"), ("mean_f1", "Mean F1"),
                        ("mean_bal_acc", "Mean BalAcc")]:
        b, a = orig_test_m[key], new_test_m[key]
        print(f"  Test {label:<19} {b:>8.2f}%  {a:>8.2f}%  {a-b:>+7.2f}%")
    print("-" * 62)
    for name, b, a in [("MIA-L", orig_mia["mia_l"], new_mia["mia_l"]),
                        ("MIA-E", orig_mia["mia_e"], new_mia["mia_e"])]:
        print(f"  {name:<24} {b*100:>8.2f}%  {a*100:>8.2f}%  "
              f"{(a-b)*100:>+7.2f}%   (ideal 50%%)")
    print(f"  {'SISA train time':<24} {'—':>9}  {sisa_train_time:>8.0f}s")
    print(f"  {'Unlearn time':<24} {'—':>9}  {unlearn_time:>8.0f}s")
    print(f"  {'Shards retrained':<24} {'—':>9}  {len(affected):>5}/{num_shards}")
    print(f"{'='*62}\n")

    _write_results(results_path, {
        "status":            "complete",
        "run_tag":           f"sisa_f{frac}_s{cfg['seed']}",
        "seed":              cfg["seed"],
        "forget_fraction":   frac,
        "forget_size":       len(forget_idx),
        "retain_size":       len(retain_idx),
        "forget_identities": forget_ids,
        "method":            "sisa",
        "num_shards":        num_shards,
        "num_slices":        num_slices,
        "affected_shards":   affected,
        "unlearn_time_s":    unlearn_time,
        "sisa_train_time_s": sisa_train_time,
        "before":            _flatten(orig_test_m, orig_retain_m, orig_forget_m, orig_mia),
        "after":             _flatten(new_test_m,  new_retain_m,  new_forget_m,  new_mia),
        "shard_stats":       all_stats,
    })
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
