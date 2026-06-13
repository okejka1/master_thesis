"""
unlearn_grad_tau.py — ∇τ (Gradient-based and Task-Agnostic) Unlearning.

Implements the algorithm from:
"∇τ: Gradient-based and Task-Agnostic machine Unlearning" (Trippa et al., 2024)
https://arxiv.org/abs/2403.14339

This is an approximate unlearning method. It modifies a pre-trained model in-place
using a custom loss that performs gradient ascent on the forget set and gradient
descent on the retain set. It uses a validation set to cap the forget loss.

Usage
-----
# Locally:
    python unlearn_grad_tau.py --config configs/cifar10.yaml

CLI overrides:
    --checkpoint-dir PATH
    --data-root PATH
    ...
"""

import argparse
import itertools
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import torchvision
from torch.utils.data import DataLoader, Subset

from models import build_resnet18
from utils import (
    class_subset_loader,
    evaluate,
    forget_sample_confidences,
    get_datasets,
    get_test_transform,
    load_checkpoint,
    per_class_accuracy,
    save_checkpoint,
    set_seed,
)
from mia import run_mia_suite


# config

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="∇τ Gradient-based machine unlearning."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--original-ckpt",  default=None)
    parser.add_argument("--data-root",      default=None)
    parser.add_argument("--forget-strategy", default=None,
                        choices=["random", "class"])
    parser.add_argument("--forget-fraction", type=float, default=None)
    parser.add_argument("--forget-class",    type=int,   default=None)
    parser.add_argument("--batch-size",      type=int,   default=None)
    parser.add_argument("--seed",            type=int,   default=None)
    parser.add_argument("--grad-tau-epochs", type=int, default=None)
    parser.add_argument("--grad-tau-alpha", type=float, default=None)
    parser.add_argument("--grad-tau-lr", type=float, default=None)
    
    return parser.parse_args()


def merge(cfg: dict, args) -> dict:
    """Overlay non-None CLI args on top of YAML config values."""
    overrides = {
        "checkpoint_dir":  args.checkpoint_dir,
        "data_root":       args.data_root,
        "forget_strategy": args.forget_strategy,
        "forget_fraction": args.forget_fraction,
        "forget_class":    args.forget_class,
        "batch_size":      args.batch_size,
        "seed":            args.seed,
        "grad_tau_forget_epochs": args.grad_tau_epochs,
        "grad_tau_alpha":         args.grad_tau_alpha,
        "grad_tau_lr":            args.grad_tau_lr,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


# splits

def build_indices(train_dataset,
                  strategy: str,
                  forget_fraction: float,
                  forget_class: int,
                  val_fraction: float,
                  seed: int) -> tuple[list, list, list]:
    """
    Partition train indices into (forget, retain_train, val).
    The validation set is carved out of the retain set.
    """
    all_indices = list(range(len(train_dataset)))
    rng = random.Random(seed)

    if strategy == "random":
        forget_indices = rng.sample(all_indices,
                                    int(len(all_indices) * forget_fraction))
    elif strategy == "class":
        forget_indices = [i for i, (_, label) in enumerate(train_dataset)
                          if label == forget_class]
    else:
        raise ValueError(f"Unknown forget strategy: {strategy!r}")

    forget_set = set(forget_indices)
    retain_indices = [i for i in all_indices if i not in forget_set]

    num_val = int(len(retain_indices) * val_fraction)
    val_indices = rng.sample(retain_indices, num_val)
    val_set = set(val_indices)
    retain_train_indices = [i for i in retain_indices if i not in val_set]

    return forget_indices, retain_train_indices, val_indices



@torch.no_grad()
def compute_mean_loss(model, loader, criterion, device) -> float:
    """Compute mean loss over a dataset (used for L_Dv)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / total_samples


# unlearning loop

def grad_tau_unlearn(model, forget_loader, retain_loader, ref_loader,
                     alpha_init: float, forget_epochs: int,
                     lr: float, weight_decay: float,
                     recompute_val_every: int, device):
    """
    ∇τ unlearning procedure (Algorithm 1 from the paper).
    Modifies `model` in-place.

    Parameters
    ----------
    ref_loader : DataLoader
        Used to compute τ = L_Dv, the TARGET loss the forget set should reach.
        Must be a loader whose loss is HIGHER than the forget set at the start
        of unlearning — i.e. data the model was NOT trained on (test set).
        Using retain training data here is wrong: the model has low loss on
        both sets, so ReLU(L_Dv − L_Df) ≈ 0 and the ascent term never fires.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_reduce = nn.CrossEntropyLoss()

    model.train()

    # Infinite iterator for retain set to match forget set steps
    def retain_generator():
        while True:
            for batch in retain_loader:
                yield batch
    retain_iter = retain_generator()

    steps_per_epoch = len(forget_loader)
    total_steps     = forget_epochs * steps_per_epoch

    alpha    = alpha_init
    step     = 0
    tau      = 0.0   # τ in the paper — recomputed every recompute_val_every epochs

    print(f"\nStarting ∇τ Unlearning — {forget_epochs} epochs  "
          f"({total_steps} steps)  α₀={alpha_init:.4f}")
    print(f"  τ source : {ref_loader.dataset.__class__.__name__} "
          f"({len(ref_loader.dataset):,} samples)\n")

    for epoch in range(1, forget_epochs + 1):
        t0 = time.time()

        if (epoch - 1) % recompute_val_every == 0:
            tau = compute_mean_loss(model, ref_loader, criterion_reduce, device)
            model.train()
            print(f"  [Epoch {epoch}] τ (ref loss) = {tau:.4f}")

        epoch_forget_loss = 0.0
        epoch_retain_loss = 0.0
        epoch_relu_active = 0      # how many batches had ReLU > 0

        for Xf, Yf in forget_loader:
            Xf, Yf = Xf.to(device), Yf.to(device)
            Xr, Yr = next(retain_iter)
            Xr, Yr = Xr.to(device), Yr.to(device)

            optimizer.zero_grad()

            # L_Df computed on EVAL-transformed images so it's comparable to τ
            # (τ is computed on eval images; using augmented images here would
            #  inflate L_Df above τ and kill the ReLU every batch)
            out_f = model(Xf)
            L_Df  = criterion_reduce(out_f, Yf)

            out_r = model(Xr)
            L_Dr  = criterion_reduce(out_r, Yr)

            # L = α · ReLU(τ − L_Df)² + (1 − α) · L_Dr
            diff      = tau - L_Df          # τ is a Python float → no graph through it
            relu_diff = F.relu(diff)
            loss = alpha * relu_diff ** 2 + (1.0 - alpha) * L_Dr

            loss.backward()
            optimizer.step()

            epoch_forget_loss += L_Df.item()
            epoch_retain_loss += L_Dr.item()
            if relu_diff.item() > 0:
                epoch_relu_active += 1

            step  += 1
            alpha  = alpha_init * max(0.0, 1.0 - step / total_steps)

        elapsed = time.time() - t0
        relu_pct = 100 * epoch_relu_active / steps_per_epoch
        print(f"  Epoch {epoch:>3} | α={alpha:.4f} | "
              f"L_Df={epoch_forget_loss/steps_per_epoch:.4f} | "
              f"L_Dr={epoch_retain_loss/steps_per_epoch:.4f} | "
              f"ReLU active={relu_pct:.0f}% | {elapsed:.1f}s")

    return model


# entry point

def _write_results(path: str, payload: dict) -> None:
    """Atomically write results JSON (write to tmp then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    args = parse_args()
    cfg  = merge(load_config(args.config), args)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    dataset     = cfg["dataset"]
    ds_tag      = dataset.lower()

    # Resolve original checkpoint path
    original_ckpt = (args.original_ckpt
                     or os.path.join(cfg["checkpoint_dir"],
                                     f"resnet18_{ds_tag}_best.pth"))

    results_path = os.path.join(cfg["checkpoint_dir"], f"grad_tau_{ds_tag}_results.json")

    if os.path.exists(results_path):
        try:
            with open(results_path) as _f:
                _existing = json.load(_f)
        except Exception:
            _existing = {}
        if _existing.get("status") == "complete":
            print(f"[grad_tau] Results already complete → {results_path}")
            return

    print(f"{'='*65}")
    print(f"  ∇τ Unlearning — {dataset}")
    print(f"  Device          : {device}")
    print(f"  Forget strategy : {cfg['forget_strategy']}", end="")
    if cfg["forget_strategy"] == "random":
        print(f"  (fraction={cfg['forget_fraction']})")
    else:
        print(f"  (class={cfg['forget_class']})")
    print(f"  Original model  : {original_ckpt}")
    print(f"  Checkpoints     : {cfg['checkpoint_dir']}")
    print(f"{'='*65}\n")

    # full_train   — augmented transforms   → used for unlearning (retain data)
    # full_eval    — no augmentation        → used for evaluation/forget ascent
    full_train, test_ds = get_datasets(dataset, cfg["data_root"])

    DatasetClass = (torchvision.datasets.CIFAR10 if dataset == "CIFAR10"
                    else torchvision.datasets.CIFAR100)
    full_eval = DatasetClass(root=cfg["data_root"], train=True,
                             download=True,
                             transform=get_test_transform(dataset))

    forget_indices, retain_indices, _ = build_indices(
        full_train,
        strategy=cfg["forget_strategy"],
        forget_fraction=cfg["forget_fraction"],
        forget_class=cfg["forget_class"],
        val_fraction=0.0,          # no val split needed — we use test set as τ reference
        seed=cfg["seed"],
    )

    print(f"Forget set size : {len(forget_indices):,}")
    print(f"Retain set size : {len(retain_indices):,}")
    print(f"Test   set size : {len(test_ds):,}\n")

    bs = cfg["batch_size"]

    # forget_train_loader: EVAL transforms — must match τ scale.
    #   (Using augmented images here inflates L_Df above τ, zeroing the ReLU.)
    forget_train_loader = DataLoader(Subset(full_eval, forget_indices),
                                     batch_size=bs, shuffle=True,
                                     num_workers=2, pin_memory=True)
    retain_train_loader = DataLoader(Subset(full_train, retain_indices),
                                     batch_size=bs, shuffle=True,
                                     num_workers=2, pin_memory=True, drop_last=True)

    forget_eval_loader     = DataLoader(Subset(full_eval, forget_indices),
                                        batch_size=bs, shuffle=False,
                                        num_workers=2, pin_memory=True)
    all_retain_eval_loader = DataLoader(Subset(full_eval, retain_indices),
                                        batch_size=bs, shuffle=False,
                                        num_workers=2, pin_memory=True)
    test_loader            = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                        num_workers=2, pin_memory=True)
    mia_test_loader = (class_subset_loader(test_ds, cfg["forget_class"], bs)
                       if cfg["forget_strategy"] == "class" else test_loader)

    # τ = mean loss on data the original model never trained on.
    # We use a fixed 1 000-sample subset of the test set so that:
    #   • the full 10 000-sample test_loader is untouched for fair evaluation
    #   • τ computation is fast (< 1 s per epoch)
    # All other methods also evaluate on the full test_loader — no asymmetry.
    import random as _random
    _rng_ref = _random.Random(cfg["seed"])
    _ref_indices = _rng_ref.sample(range(len(test_ds)), min(1000, len(test_ds)))
    ref_loader = DataLoader(Subset(test_ds, _ref_indices), batch_size=bs,
                            shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    if not os.path.exists(original_ckpt):
        raise FileNotFoundError(f"Original checkpoint not found at {original_ckpt}. "
                                "Run train.py first.")

    print("Loading original pre-trained model...")
    model = load_checkpoint(original_ckpt, device)

    print("\nEvaluating original model on all splits...")
    orig_retain_loss, orig_retain_acc = evaluate(model, all_retain_eval_loader, criterion, device)
    orig_forget_loss, orig_forget_acc = evaluate(model, forget_eval_loader,     criterion, device)
    orig_test_loss,   orig_test_acc   = evaluate(model, test_loader,            criterion, device)

    print(f"\n{'Split':<15} {'Loss':>8}  {'Accuracy':>9}")
    print("-" * 36)
    print(f"{'Retain (full)':<15} {orig_retain_loss:>8.4f}  {orig_retain_acc:>8.2f}%")
    print(f"{'Forget':<15} {orig_forget_loss:>8.4f}  {orig_forget_acc:>8.2f}%")
    print(f"{'Test':<15} {orig_test_loss:>8.4f}  {orig_test_acc:>8.2f}%")

    print("\nMIA evaluation on original model (5-fold CV):")
    orig_mia = run_mia_suite(model, forget_eval_loader, mia_test_loader, device,
                             label="Original", seed=cfg["seed"])

    num_classes = 10 if dataset == "CIFAR10" else 100
    print("\nPer-class accuracy & forget-sample confidences (original model)...")
    orig_per_class_test   = per_class_accuracy(model, test_loader,            num_classes, device).tolist()
    orig_per_class_retain = per_class_accuracy(model, all_retain_eval_loader, num_classes, device).tolist()
    orig_per_class_forget = per_class_accuracy(model, forget_eval_loader,     num_classes, device).tolist()
    orig_forget_conf      = forget_sample_confidences(model, forget_eval_loader, device)

    unlearn_start = time.time()

    forget_epochs       = cfg.get("grad_tau_forget_epochs", 10)
    lr                  = cfg.get("grad_tau_lr", 1e-4)
    weight_decay        = cfg.get("grad_tau_weight_decay", 1e-4)
    recompute_val_every = cfg.get("grad_tau_recompute_val_every", 1)

    # Paper rule of thumb (§6.4): α₀ ≈ (5/3) × |D_f| / |D_train|
    # Config value must be a positive float to override; None/null → auto-compute.
    _auto_alpha      = (5.0 / 3.0) * (len(forget_indices) / len(full_train))
    _cfg_alpha       = cfg.get("grad_tau_alpha")
    alpha_init       = float(_cfg_alpha) if isinstance(_cfg_alpha, (int, float)) else _auto_alpha
    print(f"α₀ = {alpha_init:.4f}  ({'config override' if isinstance(_cfg_alpha, (int, float)) else 'auto'}, "
          f"forget_fraction={len(forget_indices)/len(full_train):.3f})")

    model = grad_tau_unlearn(
        model,
        forget_loader=forget_train_loader,
        retain_loader=retain_train_loader,
        ref_loader=ref_loader,         # τ = 1 000-sample test subset, full test set untouched
        alpha_init=alpha_init,
        forget_epochs=forget_epochs,
        lr=lr,
        weight_decay=weight_decay,
        recompute_val_every=recompute_val_every,
        device=device,
    )

    unlearn_time = time.time() - unlearn_start
    print(f"\nUnlearning complete in {unlearn_time:.1f} seconds.\n")

    print("Evaluating unlearned model on all splits...")
    new_retain_loss, new_retain_acc = evaluate(model, all_retain_eval_loader, criterion, device)
    new_forget_loss, new_forget_acc = evaluate(model, forget_eval_loader,     criterion, device)
    new_test_loss,   new_test_acc   = evaluate(model, test_loader,            criterion, device)

    print("\nMIA evaluation on unlearned model (5-fold CV):")
    new_mia = run_mia_suite(model, forget_eval_loader, mia_test_loader, device,
                            label="Unlearned", seed=cfg["seed"])

    print("\nPer-class accuracy & forget-sample confidences (unlearned model)...")
    new_per_class_test   = per_class_accuracy(model, test_loader,            num_classes, device).tolist()
    new_per_class_retain = per_class_accuracy(model, all_retain_eval_loader, num_classes, device).tolist()
    new_per_class_forget = per_class_accuracy(model, forget_eval_loader,     num_classes, device).tolist()
    new_forget_conf      = forget_sample_confidences(model, forget_eval_loader, device)

    print(f"\n{'='*68}")
    print(f"{'Metric':<22} {'Original':>12}  {'∇τ Unlearned':>14}  {'Δ':>6}")
    print(f"{'='*68}")

    acc_rows = [
        ("Retain Accuracy", orig_retain_acc, new_retain_acc),
        ("Forget Accuracy", orig_forget_acc, new_forget_acc),
        ("Test Accuracy",   orig_test_acc,   new_test_acc),
    ]
    for name, before, after in acc_rows:
        delta = after - before
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<22} {before:>11.2f}%  {after:>13.2f}%  {sign}{delta:>5.2f}%")

    mia_rows = [
        ("MIA_L (loss)",    orig_mia["mia_l"], new_mia["mia_l"]),
        ("MIA_E (entropy)", orig_mia["mia_e"], new_mia["mia_e"]),
    ]
    print("-" * 68)
    for name, before, after in mia_rows:
        delta = after - before
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<22} {before*100:>11.2f}%  {after*100:>13.2f}%  "
              f"{sign}{delta*100:>5.2f}%   (ideal: 50%)")

    print(f"{'='*68}")

    out_ckpt = os.path.join(cfg["checkpoint_dir"], f"resnet18_{ds_tag}_grad_tau_unlearn.pth")
    save_checkpoint(
        model, out_ckpt,
        epoch=forget_epochs,
        test_acc=new_test_acc,
        dataset=dataset,
        num_classes=10 if dataset == "CIFAR10" else 100,
        extra={
            "unlearning_method": "grad_tau",
            "forget_strategy":   cfg["forget_strategy"],
            "forget_size":       len(forget_indices),
            "retain_size":       len(retain_indices),
            "config":            cfg,
            "unlearn_time_s":    unlearn_time
        }
    )

    _write_results(results_path, {
        "status":          "complete",
        "dataset":         dataset,
        "seed":            cfg["seed"],
        "forget_fraction": cfg.get("forget_fraction"),
        "method":          "grad_tau",
        "forget_strategy": cfg["forget_strategy"],
        "forget_size":     len(forget_indices),
        "retain_size":     len(retain_indices),
        "unlearn_time_s":  unlearn_time,
        "alpha_init":      alpha_init,
        "forget_epochs":   forget_epochs,
        "tau_source":      "test_set",
        "before": {
            "test_acc":             orig_test_acc,
            "retain_acc":           orig_retain_acc,
            "forget_acc":           orig_forget_acc,
            "mia_l":                orig_mia["mia_l"],
            "mia_e":                orig_mia["mia_e"],
            "per_class_acc_test":   orig_per_class_test,
            "per_class_acc_retain": orig_per_class_retain,
            "per_class_acc_forget": orig_per_class_forget,
            "forget_conf":          orig_forget_conf,
        },
        "after": {
            "test_acc":             new_test_acc,
            "retain_acc":           new_retain_acc,
            "forget_acc":           new_forget_acc,
            "mia_l":                new_mia["mia_l"],
            "mia_e":                new_mia["mia_e"],
            "per_class_acc_test":   new_per_class_test,
            "per_class_acc_retain": new_per_class_retain,
            "per_class_acc_forget": new_per_class_forget,
            "forget_conf":          new_forget_conf,
        },
    })
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
