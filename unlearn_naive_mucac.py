"""
unlearn_naive_mucac.py — Naive (retrain-from-scratch) unlearning on MUCAC.

D_f is defined at the identity level: identities are sampled randomly until
|D_f| >= forget_fraction * |D_train|.  This preserves the paper's guarantee
that no identity appears in both D_f and D_r, while enabling the fraction-based
experiment design from the plan (0.1%, 1%, 5%, 10%).

Usage:
    python unlearn_naive_mucac.py --config configs/mucac.yaml \\
        --forget-fraction 0.01 --seed 42

    # 5 replicas of the same fraction (different identity draws):
    for seed in 0 1 2 3 4; do
        python unlearn_naive_mucac.py --config configs/mucac.yaml \\
            --forget-fraction 0.01 --seed $seed
    done
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
    evaluate_multilabel,
    LABEL_COLS,
)
from mia import run_mia_suite_multilabel
from utils import set_seed



# config

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",          required=True)
    p.add_argument("--forget-fraction", type=float, default=None,
                   help="Fraction of train set to forget (e.g. 0.01 = 1%%).")
    p.add_argument("--checkpoint-dir",  default=None)
    p.add_argument("--data-root",       default=None)
    p.add_argument("--base-ckpt",       default=None,
                   help="Path to trained baseline .pth (default: "
                        "checkpoint_dir/mucac_best.pth).")
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch-size",      type=int,   default=None)
    p.add_argument("--seed",            type=int,   default=None)
    return p.parse_args()


def merge(cfg: dict, args) -> dict:
    overrides = {
        "forget_fraction": args.forget_fraction,
        "checkpoint_dir":  args.checkpoint_dir,
        "data_root":       args.data_root,
        "num_epochs":      args.epochs,
        "batch_size":      args.batch_size,
        "seed":            args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg



# forget / retain split

def build_forget_retain_indices(
    df,
    forget_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Sample identities at random until |D_f| >= forget_fraction * |D_train|.

    Returns
    -------
    (forget_indices, retain_indices, forget_identity_ids)
        forget_indices  : list of integer DataFrame row indices
        retain_indices  : list of integer DataFrame row indices
        forget_identity_ids : list of identity IDs selected for forgetting
    """
    rng      = np.random.RandomState(seed)
    target   = int(np.ceil(len(df) * forget_fraction))

    identities = df["identity"].unique().copy()
    rng.shuffle(identities)

    forget_ids, count = [], 0
    for identity in identities:
        n = int((df["identity"] == identity).sum())
        forget_ids.append(int(identity))
        count += n
        if count >= target:
            break

    mask        = df["identity"].isin(forget_ids)
    forget_idx  = df.index[mask].tolist()
    retain_idx  = df.index[~mask].tolist()
    return forget_idx, retain_idx, forget_ids



# training loop

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

    mean_acc = float(correct.sum() / (total * len(LABEL_COLS)) * 100)
    return total_loss / total, mean_acc



# helpers

def _save_ckpt(model, path, epoch, metrics, cfg, history):
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "num_labels":   len(LABEL_COLS),
        "label_cols":   LABEL_COLS,
        "dataset":      "MUCAC",
        "test_metrics": metrics,
        "config":       cfg,
        "history":      history,
    }, path)
    print(f"  → {path}")


def _load_model(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = build_resnet18_multilabel(ckpt["num_labels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _write_results(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)



# entry point

def main():
    args = parse_args()
    cfg  = merge(load_config(args.config), args)

    frac = cfg["forget_fraction"]
    if frac is None:
        raise ValueError("--forget-fraction must be set (e.g. 0.01)")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    ckpt_path = os.path.join(cfg["checkpoint_dir"], "naive_mucac_best.pth")
    json_path = os.path.join(cfg["checkpoint_dir"], "naive_mucac_results.json")
    base_ckpt = (args.base_ckpt
                 or os.path.join(cfg["checkpoint_dir"], "mucac_best.pth"))

    _skip_training = False
    _partial: dict = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                _partial = json.load(f)
        except Exception:
            _partial = {}
        if _partial.get("status") == "complete":
            print(f"[naive-mucac] Already complete → {json_path}")
            return
        if (_partial.get("status") == "training_complete"
                and os.path.exists(ckpt_path)):
            print("[naive-mucac] Training done; resuming from eval+MIA step.")
            _skip_training = True

    print(f"{'='*62}")
    print(f"  Naive Unlearning — MUCAC (identity-level D_f)")
    print(f"  Device          : {device}")
    print(f"  Forget fraction : {frac*100:.2f}%   seed={cfg['seed']}")
    print(f"  Baseline model  : {base_ckpt}")
    print(f"  Output dir      : {cfg['checkpoint_dir']}")
    print(f"{'='*62}\n")

    base = os.path.join(cfg["data_root"], "mucac_dataset")

    # train_aug: for retraining D_r
    # train_eval: same images, no augmentation → for evaluating A(D_r), A(D_f)
    train_aug  = MuCACDataset(base + "/train.csv", base + "/train",
                              transform=_train_transform())
    train_eval = MuCACDataset(base + "/train.csv", base + "/train",
                              transform=_eval_transform())
    test_ds    = MuCACDataset(base + "/test.csv",  base + "/test",
                              transform=_eval_transform())

    forget_idx, retain_idx, forget_ids = build_forget_retain_indices(
        train_aug.df, forget_fraction=frac, seed=cfg["seed"]
    )
    print(f"  Forget: {len(forget_idx):,} samples  ({len(forget_ids)} identities)")
    print(f"  Retain: {len(retain_idx):,} samples")
    print(f"  Test  : {len(test_ds):,} samples\n")

    bs = cfg["batch_size"]
    nw = cfg["num_workers"]

    retain_train_loader = DataLoader(Subset(train_aug,  retain_idx),
                                     batch_size=bs, shuffle=True,
                                     num_workers=nw, pin_memory=True)
    retain_eval_loader  = DataLoader(Subset(train_eval, retain_idx),
                                     batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=True)
    forget_loader       = DataLoader(Subset(train_eval, forget_idx),
                                     batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=True)
    test_loader         = DataLoader(test_ds,
                                     batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()

    print("Loading baseline model...")
    orig_model = _load_model(base_ckpt, device)

    print("\nBaseline metrics on all splits:")
    orig_test_m   = evaluate_multilabel(orig_model, test_loader,        device)
    orig_retain_m = evaluate_multilabel(orig_model, retain_eval_loader, device)
    orig_forget_m = evaluate_multilabel(orig_model, forget_loader,      device)

    _print_split_metrics("Test",   orig_test_m)
    _print_split_metrics("Retain", orig_retain_m)
    _print_split_metrics("Forget", orig_forget_m)

    print("\nMIA on baseline model:")
    orig_mia = run_mia_suite_multilabel(orig_model, forget_loader, test_loader,
                                        device, label="Baseline",
                                        seed=cfg["seed"])

    if not _skip_training:
        print(f"\nRetraining from scratch on D_r "
              f"({len(retain_idx):,} samples, {cfg['num_epochs']} epochs)...\n")

        naive_model = build_resnet18_multilabel(len(LABEL_COLS)).to(device)
        optimizer   = optim.Adam(naive_model.parameters(),
                                 lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler   = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"]
        )

        history   = {"train_loss": [], "train_acc": [], "test_metrics": []}
        best_bal  = 0.0

        print(f"{'Ep':>3}  {'LR':>8}  {'TrLoss':>7}  {'TrAcc':>6}  "
              f"{'TeAcc':>6}  {'TeF1':>6}  {'TeBal':>6}  {'Time':>5}")
        print("-" * 60)

        t_start = time.time()
        for epoch in range(1, cfg["num_epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                naive_model, retain_train_loader, criterion, optimizer, device
            )
            te_m = evaluate_multilabel(naive_model, test_loader, device)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["test_metrics"].append(te_m)

            lr = scheduler.get_last_lr()[0]
            print(f"{epoch:>3}  {lr:>8.2e}  {tr_loss:>7.4f}  {tr_acc:>5.1f}%  "
                  f"{te_m['mean_acc']:>5.1f}%  {te_m['mean_f1']:>5.1f}%  "
                  f"{te_m['mean_bal_acc']:>5.1f}%  {time.time()-t0:>4.0f}s")

            if te_m["mean_bal_acc"] > best_bal:
                best_bal = te_m["mean_bal_acc"]
                _save_ckpt(naive_model, ckpt_path, epoch, te_m, cfg, history)

        unlearn_time = time.time() - t_start
        print(f"\nRetrain done in {unlearn_time:.0f}s  |  best bal_acc={best_bal:.2f}%")

        _write_results(json_path, {
            "status":          "training_complete",
            "run_tag":         f"naive_f{frac}_s{cfg['seed']}",
            "seed":            cfg["seed"],
            "forget_fraction": frac,
            "forget_size":     len(forget_idx),
            "retain_size":     len(retain_idx),
            "forget_identities": forget_ids,
            "method":          "naive_retrain",
            "unlearn_time_s":  unlearn_time,
            "before":          _flatten("before", orig_test_m, orig_retain_m,
                                        orig_forget_m, orig_mia),
            "after":           None,
        })
        print(f"  Phase-1 written → {json_path}")
    else:
        unlearn_time = _partial.get("unlearn_time_s", 0.0)

    print("\nLoading best naive model...")
    naive_model = _load_model(ckpt_path, device)

    print("Evaluating naive model on all splits:")
    new_test_m   = evaluate_multilabel(naive_model, test_loader,        device)
    new_retain_m = evaluate_multilabel(naive_model, retain_eval_loader, device)
    new_forget_m = evaluate_multilabel(naive_model, forget_loader,      device)

    _print_split_metrics("Test",   new_test_m)
    _print_split_metrics("Retain", new_retain_m)
    _print_split_metrics("Forget", new_forget_m)

    print("\nMIA on naive-retrained model:")
    new_mia = run_mia_suite_multilabel(naive_model, forget_loader, test_loader,
                                       device, label="Naive",
                                       seed=cfg["seed"])

    print(f"\n{'='*62}")
    print(f"  {'Metric':<24} {'Before':>9}  {'After':>9}  {'Δ':>7}")
    print(f"{'='*62}")
    for key, label in [
        ("mean_acc",     "Mean Acc"),
        ("mean_f1",      "Mean F1"),
        ("mean_bal_acc", "Mean BalAcc"),
    ]:
        b = orig_test_m[key]
        a = new_test_m[key]
        print(f"  Test {label:<19} {b:>8.2f}%  {a:>8.2f}%  {a-b:>+7.2f}%")
    print("-" * 62)
    for name, b, a in [
        ("MIA-L", orig_mia["mia_l"], new_mia["mia_l"]),
        ("MIA-E", orig_mia["mia_e"], new_mia["mia_e"]),
    ]:
        print(f"  {name:<24} {b*100:>8.2f}%  {a*100:>8.2f}%  {(a-b)*100:>+7.2f}%"
              f"   (ideal 50%)")
    print(f"  {'Unlearn time':<24} {'—':>9}  {unlearn_time:>8.0f}s  {'1.0×':>7}")
    print(f"{'='*62}\n")

    _write_results(json_path, {
        "status":            "complete",
        "run_tag":           f"naive_f{frac}_s{cfg['seed']}",
        "seed":              cfg["seed"],
        "forget_fraction":   frac,
        "forget_size":       len(forget_idx),
        "retain_size":       len(retain_idx),
        "forget_identities": forget_ids,
        "method":            "naive_retrain",
        "unlearn_time_s":    unlearn_time,
        "speedup":           1.0,
        "before":            _flatten("before", orig_test_m, orig_retain_m,
                                      orig_forget_m, orig_mia),
        "after":             _flatten("after",  new_test_m,  new_retain_m,
                                      new_forget_m,  new_mia),
    })
    print(f"Results saved → {json_path}")



def _print_split_metrics(name: str, m: dict) -> None:
    print(f"  {name:<7}  acc={m['mean_acc']:.1f}%  f1={m['mean_f1']:.1f}%  "
          f"bal={m['mean_bal_acc']:.1f}%  "
          f"[Male f1={m['Male_f1']:.1f}%  Young f1={m['Young_f1']:.1f}%  "
          f"Smiling f1={m['Smiling_f1']:.1f}%]")


def _flatten(prefix, test_m, retain_m, forget_m, mia):
    return {
        f"test_{k}":   v for k, v in test_m.items()
    } | {
        f"retain_{k}": v for k, v in retain_m.items()
    } | {
        f"forget_{k}": v for k, v in forget_m.items()
    } | {
        "mia_l": mia["mia_l"],
        "mia_e": mia["mia_e"],
    }


if __name__ == "__main__":
    main()
