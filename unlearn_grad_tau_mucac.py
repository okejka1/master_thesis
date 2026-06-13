"""
unlearn_grad_tau_mucac.py — ∇τ unlearning for MUCAC multi-label classification.

Implements Algorithm 1 from:
"∇τ: Gradient-based and Task-Agnostic machine Unlearning" (Trippa et al., 2024)
https://arxiv.org/abs/2403.14339

Adaptations vs. the CIFAR version:
  • BCEWithLogitsLoss instead of CrossEntropyLoss
  • Identity-level D_f (same sampling as naive_mucac)
  • evaluate_multilabel for metrics (acc, f1, bal_acc per label)
  • run_mia_suite_multilabel for MIA

Usage:
    python unlearn_grad_tau_mucac.py --config configs/mucac.yaml \\
        --forget-fraction 0.01 --seed 0 \\
        --checkpoint-dir checkpoints/seed_0/mucac/instance_wise/grad_tau/frac_01pct \\
        --base-ckpt checkpoints/seed_0/mucac/base/mucac_best.pth
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

from models import build_resnet18_multilabel
from mucac_dataset import (
    MuCACDataset,
    _eval_transform,
    _train_transform,
    evaluate_multilabel,
    LABEL_COLS,
)
from mia import run_mia_suite_multilabel
from unlearn_naive_mucac import build_forget_retain_indices, _flatten, _print_split_metrics
from utils import set_seed



# config

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",           required=True)
    p.add_argument("--forget-fraction",  type=float, default=None)
    p.add_argument("--checkpoint-dir",   default=None)
    p.add_argument("--base-ckpt",        default=None)
    p.add_argument("--data-root",        default=None)
    p.add_argument("--batch-size",       type=int,   default=None)
    p.add_argument("--seed",             type=int,   default=None)
    p.add_argument("--grad-tau-epochs",  type=int,   default=None)
    p.add_argument("--grad-tau-alpha",   type=float, default=None)
    p.add_argument("--grad-tau-lr",      type=float, default=None)
    return p.parse_args()


def merge(cfg: dict, args) -> dict:
    overrides = {
        "forget_fraction":      args.forget_fraction,
        "checkpoint_dir":       args.checkpoint_dir,
        "data_root":            args.data_root,
        "batch_size":           args.batch_size,
        "seed":                 args.seed,
        "grad_tau_forget_epochs": args.grad_tau_epochs,
        "grad_tau_alpha":         args.grad_tau_alpha,
        "grad_tau_lr":            args.grad_tau_lr,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg



# helpers

def _load_model(path: str, device: torch.device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = build_resnet18_multilabel(ckpt["num_labels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _save_ckpt(model, path: str, epoch: int, metrics: dict, cfg: dict):
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "num_labels":   len(LABEL_COLS),
        "label_cols":   LABEL_COLS,
        "dataset":      "MUCAC",
        "test_metrics": metrics,
        "config":       cfg,
    }, path)
    print(f"  → {path}")


def _write_results(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)



@torch.no_grad()
# unlearning loop

def _mean_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss, total_n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        loss = criterion(model(imgs), labels)
        total_loss += loss.item() * imgs.size(0)
        total_n    += imgs.size(0)
    return total_loss / total_n



# grad tau unlearn

def grad_tau_unlearn(model,
                     forget_loader: DataLoader,
                     retain_loader: DataLoader,
                     ref_loader:    DataLoader,
                     alpha_init:    float,
                     forget_epochs: int,
                     lr:            float,
                     weight_decay:  float,
                     recompute_val_every: int,
                     device:        torch.device):
    """
    ∇τ unlearning — Algorithm 1, adapted for BCEWithLogitsLoss.

    L = α · ReLU(τ − L_Df)² + (1 − α) · L_Dr

    τ is computed on ref_loader (unseen test data) so that L_Df < τ at the
    start and gradient ascent on D_f fires until model "forgets" D_f.
    α decays linearly to 0 over all steps, gradually shifting weight to retain.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()

    def _retain_gen():
        while True:
            for batch in retain_loader:
                yield batch
    retain_iter = _retain_gen()

    steps_per_epoch = len(forget_loader)
    total_steps     = forget_epochs * steps_per_epoch
    alpha           = alpha_init
    tau             = 0.0
    step            = 0

    print(f"\n∇τ Unlearning — {forget_epochs} epochs  "
          f"({total_steps} steps)  α₀={alpha_init:.4f}")

    for epoch in range(1, forget_epochs + 1):
        t0 = time.time()

        if (epoch - 1) % recompute_val_every == 0:
            tau = _mean_loss(model, ref_loader, criterion, device)
            model.train()
            print(f"  [Epoch {epoch}] τ (ref loss) = {tau:.4f}")

        ep_loss_f, ep_loss_r, ep_relu = 0.0, 0.0, 0

        for Xf, Yf in forget_loader:
            Xf, Yf = Xf.to(device), Yf.to(device)
            Xr, Yr = next(retain_iter)
            Xr, Yr = Xr.to(device), Yr.to(device)

            optimizer.zero_grad()

            L_Df = criterion(model(Xf), Yf)
            L_Dr = criterion(model(Xr), Yr)

            relu_diff = F.relu(tau - L_Df)
            loss = alpha * relu_diff ** 2 + (1.0 - alpha) * L_Dr

            loss.backward()
            optimizer.step()

            ep_loss_f += L_Df.item()
            ep_loss_r += L_Dr.item()
            if relu_diff.item() > 0:
                ep_relu += 1

            step  += 1
            alpha  = alpha_init * max(0.0, 1.0 - step / total_steps)

        print(f"  Epoch {epoch:>3} | α={alpha:.4f} | "
              f"L_Df={ep_loss_f/steps_per_epoch:.4f} | "
              f"L_Dr={ep_loss_r/steps_per_epoch:.4f} | "
              f"ReLU={100*ep_relu/steps_per_epoch:.0f}% | "
              f"{time.time()-t0:.1f}s")

    return model



# entry point

def main():
    args = parse_args()
    cfg  = merge(load_config(args.config), args)

    frac = cfg["forget_fraction"]
    if frac is None:
        raise ValueError("--forget-fraction required")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    ckpt_path = os.path.join(cfg["checkpoint_dir"], "grad_tau_mucac_best.pth")
    json_path = os.path.join(cfg["checkpoint_dir"], "grad_tau_mucac_results.json")
    base_ckpt = (args.base_ckpt
                 or os.path.join(cfg["checkpoint_dir"], "mucac_best.pth"))

    if os.path.exists(json_path):
        try:
            if json.load(open(json_path)).get("status") == "complete":
                print(f"[grad-tau-mucac] Already complete → {json_path}")
                return
        except Exception:
            pass

    print(f"{'='*62}")
    print(f"  ∇τ Unlearning — MUCAC")
    print(f"  Device          : {device}")
    print(f"  Forget fraction : {frac*100:.2f}%   seed={cfg['seed']}")
    print(f"  Baseline model  : {base_ckpt}")
    print(f"  Output dir      : {cfg['checkpoint_dir']}")
    print(f"{'='*62}\n")

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
    n_train = len(train_aug)

    print(f"  Forget: {len(forget_idx):,} samples  ({len(forget_ids)} identities)")
    print(f"  Retain: {len(retain_idx):,} samples")
    print(f"  Test  : {len(test_ds):,} samples\n")

    bs = cfg["batch_size"]
    nw = cfg["num_workers"]

    # forget_loader uses eval transform — matches τ scale (no augmentation jitter)
    forget_loader      = DataLoader(Subset(train_eval, forget_idx),
                                    batch_size=bs, shuffle=True,
                                    num_workers=nw, pin_memory=True)
    forget_eval_loader = DataLoader(Subset(train_eval, forget_idx),
                                    batch_size=bs, shuffle=False,
                                    num_workers=nw, pin_memory=True)
    retain_loader      = DataLoader(Subset(train_aug,  retain_idx),
                                    batch_size=bs, shuffle=True,
                                    num_workers=nw, pin_memory=True, drop_last=True)
    retain_eval_loader = DataLoader(Subset(train_eval, retain_idx),
                                    batch_size=bs, shuffle=False,
                                    num_workers=nw, pin_memory=True)
    test_loader        = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                    num_workers=nw, pin_memory=True)

    # τ reference: fixed 1000-sample test subset (unseen → high loss)
    _rng = random.Random(cfg["seed"])
    _ref_idx = _rng.sample(range(len(test_ds)), min(1000, len(test_ds)))
    ref_loader = DataLoader(Subset(test_ds, _ref_idx), batch_size=bs,
                            shuffle=False, num_workers=nw, pin_memory=True)

    print("Loading baseline model...")
    model = _load_model(base_ckpt, device)

    print("\nBaseline metrics:")
    orig_test_m   = evaluate_multilabel(model, test_loader,        device)
    orig_retain_m = evaluate_multilabel(model, retain_eval_loader, device)
    orig_forget_m = evaluate_multilabel(model, forget_eval_loader, device)
    _print_split_metrics("Test",   orig_test_m)
    _print_split_metrics("Retain", orig_retain_m)
    _print_split_metrics("Forget", orig_forget_m)

    print("\nMIA on baseline:")
    orig_mia = run_mia_suite_multilabel(model, forget_eval_loader, test_loader,
                                        device, label="Baseline", seed=cfg["seed"])

    forget_epochs       = cfg.get("grad_tau_forget_epochs", 1)
    lr                  = cfg.get("grad_tau_lr",            1e-4)
    weight_decay        = cfg.get("grad_tau_weight_decay",  1e-4)
    recompute_val_every = cfg.get("grad_tau_recompute_val_every", 1)

    # α₀ ≈ (5/3) × |D_f| / |D_train|  (Trippa et al. §6.4)
    _cfg_alpha  = cfg.get("grad_tau_alpha")
    alpha_init  = (float(_cfg_alpha)
                   if isinstance(_cfg_alpha, (int, float))
                   else (5.0 / 3.0) * (len(forget_idx) / n_train))
    print(f"\nα₀ = {alpha_init:.4f}  "
          f"({'config' if isinstance(_cfg_alpha, (int, float)) else 'auto'})")

    t0 = time.time()
    model = grad_tau_unlearn(
        model,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        ref_loader=ref_loader,
        alpha_init=alpha_init,
        forget_epochs=forget_epochs,
        lr=lr,
        weight_decay=weight_decay,
        recompute_val_every=recompute_val_every,
        device=device,
    )
    unlearn_time = time.time() - t0
    print(f"\nUnlearning done in {unlearn_time:.0f}s")

    print("\nMetrics after ∇τ:")
    new_test_m   = evaluate_multilabel(model, test_loader,        device)
    new_retain_m = evaluate_multilabel(model, retain_eval_loader, device)
    new_forget_m = evaluate_multilabel(model, forget_eval_loader, device)
    _print_split_metrics("Test",   new_test_m)
    _print_split_metrics("Retain", new_retain_m)
    _print_split_metrics("Forget", new_forget_m)

    print("\nMIA after ∇τ:")
    new_mia = run_mia_suite_multilabel(model, forget_eval_loader, test_loader,
                                       device, label="∇τ", seed=cfg["seed"])

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
              f"{(a-b)*100:>+7.2f}%   (ideal 50%)")
    print(f"{'='*62}\n")

    _save_ckpt(model, ckpt_path, forget_epochs, new_test_m, cfg)

    _write_results(json_path, {
        "status":            "complete",
        "run_tag":           f"grad_tau_f{frac}_s{cfg['seed']}",
        "seed":              cfg["seed"],
        "forget_fraction":   frac,
        "forget_size":       len(forget_idx),
        "retain_size":       len(retain_idx),
        "forget_identities": forget_ids,
        "method":            "grad_tau",
        "unlearn_time_s":    unlearn_time,
        "alpha_init":        alpha_init,
        "forget_epochs":     forget_epochs,
        "tau_source":        "test_set_1000",
        "before":            _flatten("before", orig_test_m, orig_retain_m,
                                      orig_forget_m, orig_mia),
        "after":             _flatten("after",  new_test_m,  new_retain_m,
                                      new_forget_m,  new_mia),
    })
    print(f"Results saved → {json_path}")


if __name__ == "__main__":
    main()
