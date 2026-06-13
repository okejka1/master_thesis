"""
ovr.py — One-vs-Rest ensemble training *and* unlearning, in a single script.

OVR replaces the single multi-class model with an ensemble of ``c`` BINARY
classifiers, one dedicated to each class.  Sub-model ``f_i`` answers
"does this input belong to class i?":
    positives = all samples of class i
    negatives = a balanced subset of all other classes (ovr_neg_ratio · |class i|)
Inference decodes one-vs-rest: predicted label = argmax_i σ(f_i(x)).

Architectural bet (see OVR_implementation_reference.md):
    A class's POSITIVE influence is localised in exactly one model → class-wise
    unlearning can be a single model DROP (O(1)).  Its NEGATIVE influence is
    diffuse (it was a negative in every other model) → measured, and optionally
    retrained away.

Usage
-----
# Train the ensemble (once per seed/dataset, like train_sisa.py):
    python ovr.py --mode train --config configs/cifar10.yaml

# Class-wise unlearning (drop the dedicated model — cheapest, approximate):
    python ovr.py --mode unlearn --config configs/cifar10.yaml \
                  --forget-strategy class --forget-class 3 --ovr-variant drop

# Class-wise, exact-within-architecture (also retrain survivors w/o class-k negatives):
    python ovr.py --mode unlearn --config configs/cifar10.yaml \
                  --forget-strategy class --forget-class 3 --ovr-variant drop_neg_retrain

# Sample-wise unlearning — full retrain (ovr_slices=1, default):
    python ovr.py --mode unlearn --config configs/cifar10.yaml \
                  --forget-strategy random --forget-fraction 0.01 --ovr-variant slice_resume

# Sample-wise unlearning — slice resume (requires ovr_slices>1 at train time):
    python ovr.py --mode unlearn --config configs/cifar10.yaml \
                  --forget-strategy random --forget-fraction 0.01 --ovr-variant slice_resume \
                  --ovr-slices 5

# Separate ensemble tree from results output (used by run_sweep.py):
    python ovr.py --mode unlearn --config configs/cifar10.yaml \
                  --checkpoint-dir checkpoints/seed_0/cifar10/ovr \
                  --output-dir     checkpoints/seed_0/cifar10/class_wise/ovr/class_3
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models import build_resnet18
from mia import run_mia_suite
from utils import (
    class_subset_loader,
    evaluate,
    forget_sample_confidences,
    get_datasets,
    get_test_transform,
    per_class_accuracy,
    save_checkpoint,
    set_seed,
)

# Finite stand-in for the logit of a dropped class.  exp(-30) ≈ 1e-13 → that
# class gets ~0 softmax mass (so it can never be predicted and a forgotten-class
# sample's true-label confidence collapses) while keeping every downstream loss /
# entropy / MIA computation numerically finite.
DROP_FILL = -30.0


# config

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="One-vs-Rest ensemble (un)learning.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--mode", required=True, choices=["train", "unlearn"])
    p.add_argument("--checkpoint-dir", default=None,
                   help="Root where the OVR ensemble tree lives "
                        "(ovr_<dataset>/ subdir).")
    p.add_argument("--output-dir", default=None,
                   help="Where to write unlearn_results.json. Defaults to the "
                        "ovr_<dataset>/ dir inside --checkpoint-dir.")
    p.add_argument("--data-root", default=None)
    p.add_argument("--forget-strategy", default=None, choices=["random", "class"])
    p.add_argument("--forget-fraction", type=float, default=None)
    p.add_argument("--forget-class", type=int, default=None)
    p.add_argument("--ovr-variant", default=None,
                   choices=["drop", "drop_neg_retrain", "slice_resume"])
    p.add_argument("--ovr-neg-ratio", type=float, default=None)
    p.add_argument("--ovr-slices", type=int, default=None)
    p.add_argument("--ovr-epochs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def merge(cfg: dict, args) -> dict:
    """Overlay non-None CLI args on top of YAML config values."""
    overrides = {
        "checkpoint_dir":  args.checkpoint_dir,
        "output_dir":      args.output_dir,
        "data_root":       args.data_root,
        "forget_strategy": args.forget_strategy,
        "forget_fraction": args.forget_fraction,
        "forget_class":    args.forget_class,
        "ovr_variant":     args.ovr_variant,
        "ovr_neg_ratio":   args.ovr_neg_ratio,
        "ovr_slices":      args.ovr_slices,
        "ovr_epochs":      args.ovr_epochs,
        "num_epochs":      args.epochs,
        "lr":              args.lr,
        "batch_size":      args.batch_size,
        "seed":            args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def num_classes_of(dataset: str) -> int:
    return 10 if dataset == "CIFAR10" else 100


def ovr_epochs_of(cfg: dict) -> int:
    return cfg.get("ovr_epochs") or cfg["num_epochs"]


# binary sub-model

def build_ovr_binary() -> nn.Module:
    """ResNet-18 (CIFAR stem) with a single-logit head for binary one-vs-rest."""
    return build_resnet18(num_classes=1)


def load_binary(path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    m = build_ovr_binary().to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


# positives / negatives

def build_pos_neg(targets: np.ndarray,
                  num_classes: int,
                  neg_ratio: float,
                  seed: int) -> dict[int, dict[str, list[int]]]:
    """
    For each class i build {"pos": [...], "neg": [...]}.

    Positives = all indices of class i.
    Negatives = round(neg_ratio · |pos|) indices sampled UNIFORMLY across the
    other classes (so no single class dominates the negatives). Deterministic.
    """
    targets = np.asarray(targets)
    by_class = {c: np.where(targets == c)[0] for c in range(num_classes)}
    rng = np.random.RandomState(seed)

    assignment: dict[int, dict[str, list[int]]] = {}
    for i in range(num_classes):
        pos = by_class[i]
        n_neg = int(round(neg_ratio * len(pos)))
        others = [c for c in range(num_classes) if c != i]

        # Even split of the negative budget across the other classes.
        per_class = max(1, n_neg // len(others))
        neg: list[int] = []
        for c in others:
            pool = by_class[c]
            take = min(per_class, len(pool))
            neg.extend(rng.choice(pool, size=take, replace=False).tolist())

        # Trim / top-up to hit n_neg exactly.
        rng.shuffle(neg)
        if len(neg) > n_neg:
            neg = neg[:n_neg]
        elif len(neg) < n_neg:
            remaining = list(set(np.concatenate([by_class[c] for c in others]).tolist())
                             - set(neg))
            extra = rng.choice(remaining,
                               size=min(n_neg - len(neg), len(remaining)),
                               replace=False).tolist()
            neg.extend(extra)

        assignment[i] = {"pos": pos.tolist(), "neg": neg}
    return assignment


# binary dataset

class BinaryView(Dataset):
    """Wraps a base dataset, yielding (image, binary_label_float) for the given
    global indices and matching labels."""

    def __init__(self, base, indices: list[int], labels: list[float]):
        assert len(indices) == len(labels)
        self.base = base
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        img, _ = self.base[self.indices[k]]
        return img, torch.tensor(self.labels[k], dtype=torch.float32)


def balanced_loader(base, pos: list[int], neg: list[int],
                    batch_size: int) -> DataLoader:
    """DataLoader over pos∪neg with a sampler that draws ~50/50 each batch."""
    indices = list(pos) + list(neg)
    labels = [1.0] * len(pos) + [0.0] * len(neg)
    ds = BinaryView(base, indices, labels)

    w_pos = 0.5 / max(1, len(pos))
    w_neg = 0.5 / max(1, len(neg))
    weights = [w_pos] * len(pos) + [w_neg] * len(neg)
    sampler = WeightedRandomSampler(weights, num_samples=len(indices),
                                    replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=2, pin_memory=True)


# training

def train_binary_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def train_binary_model(class_id: int,
                       pos: list[int],
                       neg: list[int],
                       base_train,
                       cfg: dict,
                       device: torch.device,
                       save_path: str | None = None,
                       verbose: bool = True) -> nn.Module:
    """Train one binary sub-model with optional SISA-style slicing.

    When ovr_slices > 1, pos/neg are split into cumulative slices and a
    slice_XX.pth checkpoint (with optimizer state) is saved after each slice,
    enabling slice_resume unlearning without full retraining.
    """
    epochs = ovr_epochs_of(cfg)
    num_slices = cfg.get("ovr_slices") or 1
    epochs_per_slice = max(1, epochs // num_slices)

    model = build_ovr_binary().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg["lr"],
                          momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"], nesterov=True)
    # Single scheduler spanning all epochs — milestones are absolute epoch numbers.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"])

    pos_raw = [chunk.tolist() for chunk in np.array_split(np.array(pos), num_slices)]
    neg_raw = [chunk.tolist() for chunk in np.array_split(np.array(neg), num_slices)]

    if verbose:
        print(f"  class {class_id:>3}  |  {len(pos)} pos / {len(neg)} neg  "
              f"|  {epochs} epochs  |  {num_slices} slice(s)")

    cum_pos: list[int] = []
    cum_neg: list[int] = []
    total_epoch = 0

    for slice_id in range(num_slices):
        cum_pos += pos_raw[slice_id]
        cum_neg += neg_raw[slice_id]
        loader = balanced_loader(base_train, cum_pos, cum_neg, cfg["batch_size"])

        for epoch in range(1, epochs_per_slice + 1):
            total_epoch += 1
            loss, acc = train_binary_epoch(model, loader, criterion, optimizer, device)
            scheduler.step()
            if verbose and (total_epoch == 1 or
                            total_epoch % max(1, epochs // 5) == 0 or
                            total_epoch == epochs):
                print(f"      epoch {total_epoch:>3}  lr {scheduler.get_last_lr()[0]:.5f}  "
                      f"bce {loss:.4f}  bal-acc {acc:.2f}%")

        if save_path is not None and num_slices > 1:
            slice_dir = os.path.dirname(save_path)
            os.makedirs(slice_dir, exist_ok=True)
            torch.save({
                "epoch":       total_epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "class_id":    class_id,
                "slice_id":    slice_id,
                "num_slices":  num_slices,
            }, os.path.join(slice_dir, f"slice_{slice_id:02d}.pth"))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_checkpoint(
            model, save_path, epoch=total_epoch, test_acc=0.0,
            dataset=cfg["dataset"], num_classes=num_classes_of(cfg["dataset"]),
            extra={"class_id": class_id, "binary": True,
                   "n_pos": len(pos), "n_neg": len(neg),
                   "neg_ratio": cfg["ovr_neg_ratio"], "config": cfg},
        )
    return model


# ensemble adapter

class OvREnsemble(nn.Module):
    """Wrap c binary sub-models; forward() returns (B, c) class logits so that all
    existing multi-class eval code (utils.evaluate / per_class_accuracy /
    forget_sample_confidences and mia.run_mia_suite) works unchanged.

    Dropped (unlearned) classes are masked to DROP_FILL so they can never be
    predicted and contribute ~0 softmax mass."""

    def __init__(self, submodels: list[nn.Module], num_classes: int):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.register_buffer("active", torch.ones(num_classes, dtype=torch.bool))

    def drop(self, class_id: int):
        self.active[class_id] = False

    @property
    def active_classes(self) -> list[int]:
        return [i for i, a in enumerate(self.active.tolist()) if a]

    def forward(self, x):
        logits = torch.stack([m(x).squeeze(1) for m in self.submodels], dim=1)
        mask = ~self.active.to(logits.device).unsqueeze(0)
        return logits.masked_fill(mask, DROP_FILL)


# forget set

def build_forget_indices(targets: np.ndarray,
                         num_train: int,
                         strategy: str,
                         forget_fraction: float,
                         forget_class: int,
                         seed: int) -> list[int]:
    if strategy == "random":
        rng = random.Random(seed)
        return rng.sample(range(num_train), int(num_train * forget_fraction))
    if strategy == "class":
        return np.where(np.asarray(targets) == forget_class)[0].tolist()
    raise ValueError(f"Unknown forget strategy: {strategy!r}")


# sub-model retraining

def retrain_binary_model(class_id: int,
                         pos: list[int],
                         neg: list[int],
                         forget_set: set[int],
                         base_train,
                         cfg: dict,
                         device: torch.device) -> tuple[nn.Module, dict]:
    """Retrain sub-model `class_id` from scratch on its retained pos/neg
    (forget samples removed from BOTH roles).  Returns (model, stats)."""
    new_pos = [i for i in pos if i not in forget_set]
    new_neg = [i for i in neg if i not in forget_set]
    removed_pos = len(pos) - len(new_pos)
    removed_neg = len(neg) - len(new_neg)

    t0 = time.time()
    model = train_binary_model(class_id, new_pos, new_neg, base_train, cfg,
                               device, save_path=None, verbose=False)
    stats = {
        "class_id": class_id, "removed_pos": removed_pos,
        "removed_neg": removed_neg, "retrain_time_s": time.time() - t0,
    }
    return model, stats


def resume_binary_from_slice(class_id: int,
                              pos: list[int],
                              neg: list[int],
                              forget_set: set[int],
                              class_dir: str,
                              base_train,
                              cfg: dict,
                              device: torch.device) -> tuple[nn.Module, dict]:
    """Slice-aware retrain mirroring SISA's retrain_shard.

    Loads the checkpoint from the slice just before the first affected one,
    then retrains remaining slices without forget samples.  Falls back to
    full retrain if slice checkpoints are absent (e.g. trained with ovr_slices=1).
    """
    num_slices = cfg.get("ovr_slices") or 1
    epochs = ovr_epochs_of(cfg)
    epochs_per_slice = max(1, epochs // num_slices)

    pos_raw = [chunk.tolist() for chunk in np.array_split(np.array(pos), num_slices)]
    neg_raw = [chunk.tolist() for chunk in np.array_split(np.array(neg), num_slices)]

    first_affected = next(
        (s for s in range(num_slices)
         if forget_set & (set(pos_raw[s]) | set(neg_raw[s]))),
        num_slices,
    )

    if first_affected == 0 or num_slices == 1:
        model = build_ovr_binary().to(device)
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"],
                              momentum=cfg["momentum"],
                              weight_decay=cfg["weight_decay"], nesterov=True)
        start_epoch = 0
    else:
        prev_ckpt_path = os.path.join(class_dir, f"slice_{first_affected - 1:02d}.pth")
        if not os.path.exists(prev_ckpt_path):
            print(f"    WARNING: {prev_ckpt_path} not found — falling back to full retrain")
            return retrain_binary_model(class_id, pos, neg, forget_set,
                                        base_train, cfg, device)
        ckpt = torch.load(prev_ckpt_path, map_location=device)
        model = build_ovr_binary().to(device)
        model.load_state_dict(ckpt["model_state"])
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"],
                              momentum=cfg["momentum"],
                              weight_decay=cfg["weight_decay"], nesterov=True)
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"]
        print(f"    class {class_id}: loaded slice_{first_affected - 1:02d}.pth "
              f"(epoch {start_epoch}), retraining from slice {first_affected}")

    # Scheduler resumes from start_epoch — milestones are still absolute.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["lr_milestones"],
        gamma=cfg["lr_gamma"],
        last_epoch=start_epoch - 1,
    )
    criterion = nn.BCEWithLogitsLoss()

    cum_pos = [i for s in range(first_affected) for i in pos_raw[s] if i not in forget_set]
    cum_neg = [i for s in range(first_affected) for i in neg_raw[s] if i not in forget_set]

    total_retrain_epochs = 0
    t0 = time.time()

    for s_id in range(first_affected, num_slices):
        cum_pos += [i for i in pos_raw[s_id] if i not in forget_set]
        cum_neg += [i for i in neg_raw[s_id] if i not in forget_set]
        loader = balanced_loader(base_train, cum_pos, cum_neg, cfg["batch_size"])

        print(f"    Retraining slice {s_id}/{num_slices - 1}  "
              f"({len(cum_pos)} pos, {len(cum_neg)} neg)  "
              f"→ {epochs_per_slice} epochs")

        for epoch in range(1, epochs_per_slice + 1):
            total_retrain_epochs += 1
            loss, acc = train_binary_epoch(model, loader, criterion, optimizer, device)
            scheduler.step()
            print(f"      epoch {start_epoch + total_retrain_epochs:>3}  "
                  f"lr {scheduler.get_last_lr()[0]:.5f}  "
                  f"bce {loss:.4f}  bal-acc {acc:.2f}%")

    stats = {
        "class_id":             class_id,
        "retrained":            True,
        "first_affected_slice": first_affected,
        "slices_retrained":     num_slices - first_affected,
        "epochs_retrained":     total_retrain_epochs,
        "removed_pos":          sum(1 for i in pos if i in forget_set),
        "removed_neg":          sum(1 for i in neg if i in forget_set),
        "retrain_time_s":       time.time() - t0,
    }
    return model, stats


# ovr metrics

def negative_footprint(assignment: dict, forgotten_classes: list[int],
                       active_classes: list[int]) -> dict:
    """How much of the forgotten class(es) still lives as NEGATIVES in surviving
    sub-models after a drop."""
    forget_samples = set()
    for k in forgotten_classes:
        forget_samples.update(assignment[str(k)]["pos"])

    pair_count = 0
    touched_models = set()
    survivors_neg = {i: set(assignment[str(i)]["neg"]) for i in active_classes}
    left_as_neg = set()
    for i, negset in survivors_neg.items():
        overlap = forget_samples & negset
        if overlap:
            touched_models.add(i)
            pair_count += len(overlap)
            left_as_neg |= overlap

    n_forget = max(1, len(forget_samples))
    return {
        "neg_footprint_count": pair_count,
        "neg_footprint_frac": len(left_as_neg) / n_forget,
        "models_touched": len(touched_models),
        "n_active_models": len(active_classes),
    }


# evaluation bundle

def evaluate_snapshot(ensemble, num_classes, device,
                      test_loader, retain_loader, forget_loader, seed,
                      mia_test_loader=None):
    ce = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(ensemble, test_loader, ce, device)
    ret_loss, ret_acc = evaluate(ensemble, retain_loader, ce, device)
    fgt_loss, fgt_acc = evaluate(ensemble, forget_loader, ce, device)
    _mia_test = mia_test_loader if mia_test_loader is not None else test_loader
    mia = run_mia_suite(ensemble, forget_loader, _mia_test, device, seed=seed)
    return {
        "test_acc": test_acc, "retain_acc": ret_acc, "forget_acc": fgt_acc,
        "mia_l": mia["mia_l"], "mia_e": mia["mia_e"],
        "per_class_acc_test": per_class_accuracy(
            ensemble, test_loader, num_classes, device).tolist(),
        "per_class_acc_retain": per_class_accuracy(
            ensemble, retain_loader, num_classes, device).tolist(),
        "per_class_acc_forget": per_class_accuracy(
            ensemble, forget_loader, num_classes, device).tolist(),
        "forget_conf": forget_sample_confidences(ensemble, forget_loader, device),
    }


# results

def _write_results(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


# train mode

def run_train(cfg: dict, device: torch.device):
    dataset = cfg["dataset"]
    ds_tag = dataset.lower()
    c = num_classes_of(dataset)
    ovr_dir = os.path.join(cfg["checkpoint_dir"], f"ovr_{ds_tag}")
    os.makedirs(ovr_dir, exist_ok=True)

    print(f"{'='*65}")
    print(f"  OVR Training — {dataset}")
    print(f"  Device     : {device}")
    print(f"  Classes (c): {c}  (one binary sub-model each)")
    print(f"  neg_ratio  : {cfg['ovr_neg_ratio']}")
    print(f"  epochs/sub : {ovr_epochs_of(cfg)}")
    print(f"  Tree       : {ovr_dir}")
    print(f"{'='*65}\n")

    train_ds, test_ds = get_datasets(dataset, cfg["data_root"])
    targets = np.array(train_ds.targets)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    assignment = build_pos_neg(targets, c, cfg["ovr_neg_ratio"], cfg["seed"])
    with open(os.path.join(ovr_dir, "class_assignments.json"), "w") as f:
        json.dump({str(k): v for k, v in assignment.items()}, f)
    print(f"Class assignments saved → {ovr_dir}/class_assignments.json\n")

    t0 = time.time()
    submodels = []
    for i in range(c):
        set_seed(cfg["seed"] + i)
        m = train_binary_model(i, assignment[i]["pos"], assignment[i]["neg"],
                               train_ds, cfg, device,
                               save_path=os.path.join(ovr_dir, f"class_{i}", "model.pth"))
        m.eval()
        submodels.append(m)
    total_time = time.time() - t0

    ensemble = OvREnsemble(submodels, c).to(device)
    ce = nn.CrossEntropyLoss()
    _, ens_acc = evaluate(ensemble, test_loader, ce, device)
    print(f"\n{'='*65}")
    print(f"  Ensemble test accuracy (one-vs-rest argmax): {ens_acc:.2f}%")
    print(f"  Total training time: {total_time:.1f}s")
    print(f"{'='*65}")

    with open(os.path.join(ovr_dir, "ensemble_meta.json"), "w") as f:
        json.dump({
            "status": "complete", "dataset": dataset, "method": "ovr",
            "num_classes": c, "neg_ratio": cfg["ovr_neg_ratio"],
            "ovr_epochs": ovr_epochs_of(cfg), "ensemble_acc": ens_acc,
            "total_time_s": total_time, "config": cfg,
        }, f, indent=2, default=str)
    print(f"\nEnsemble metadata → {ovr_dir}/ensemble_meta.json")


# unlearn mode

def run_unlearn(cfg: dict, device: torch.device):
    dataset = cfg["dataset"]
    ds_tag = dataset.lower()
    c = num_classes_of(dataset)
    ovr_dir = os.path.join(cfg["checkpoint_dir"], f"ovr_{ds_tag}")

    strategy = cfg["forget_strategy"]
    variant = cfg.get("ovr_variant") or (
        "drop" if strategy == "class" else "slice_resume")
    if variant in ("drop", "drop_neg_retrain") and strategy != "class":
        raise ValueError(f"variant {variant!r} is class-wise only "
                         f"(got strategy={strategy!r}).")

    out_dir = cfg.get("output_dir") or ovr_dir
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "unlearn_results.json")

    print(f"{'='*65}")
    print(f"  OVR Unlearning — {dataset}")
    print(f"  Variant         : {variant}")
    print(f"  Forget strategy : {strategy}", end="")
    print(f"  (class={cfg['forget_class']})" if strategy == "class"
          else f"  (fraction={cfg['forget_fraction']})")
    print(f"  Tree            : {ovr_dir}")
    print(f"  Results out     : {out_dir}")
    print(f"{'='*65}\n")

    train_ds, test_ds = get_datasets(dataset, cfg["data_root"])
    targets = np.array(train_ds.targets)
    DatasetClass = (torchvision.datasets.CIFAR10 if dataset == "CIFAR10"
                    else torchvision.datasets.CIFAR100)
    full_eval = DatasetClass(root=cfg["data_root"], train=True, download=True,
                             transform=get_test_transform(dataset))
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    with open(os.path.join(ovr_dir, "class_assignments.json")) as f:
        assignment = json.load(f)  # keys are strings

    forget_indices = build_forget_indices(
        targets, len(train_ds), strategy,
        cfg["forget_fraction"], cfg["forget_class"], cfg["seed"])
    forget_set = set(forget_indices)
    retain_indices = [i for i in range(len(train_ds)) if i not in forget_set]
    print(f"Forget set : {len(forget_indices):,} samples")
    print(f"Retain set : {len(retain_indices):,} samples\n")
    forget_loader = DataLoader(torch.utils.data.Subset(full_eval, forget_indices),
                               batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=2, pin_memory=True)
    retain_loader = DataLoader(torch.utils.data.Subset(full_eval, retain_indices),
                               batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=2, pin_memory=True)
    mia_test_loader = (class_subset_loader(test_ds, cfg["forget_class"], cfg["batch_size"])
                       if strategy == "class" else None)

    print("Loading original OVR ensemble...")
    submodels = [load_binary(os.path.join(ovr_dir, f"class_{i}", "model.pth"),
                             device) for i in range(c)]
    ensemble = OvREnsemble(submodels, c).to(device)

    print("Evaluating original ensemble (this includes 2× MIA)...")
    before = evaluate_snapshot(ensemble, c, device, test_loader,
                               retain_loader, forget_loader, cfg["seed"],
                               mia_test_loader=mia_test_loader)

    print(f"\n{'='*65}\n  Unlearning — variant = {variant}\n{'='*65}")
    unlearn_start = time.time()
    shard_stats = []

    if variant == "drop":
        for k in sorted({int(targets[i]) for i in forget_indices}):
            ensemble.drop(k)
            print(f"  dropped sub-model f_{k} (O(1), no retraining)")

    elif variant == "drop_neg_retrain":
        forgotten = sorted({int(targets[i]) for i in forget_indices})
        for k in forgotten:
            ensemble.drop(k)
            print(f"  dropped sub-model f_{k}")
        survivors = ensemble.active_classes
        print(f"  retraining {len(survivors)} survivors without class-"
              f"{forgotten} negatives...")
        for i in survivors:
            pos = assignment[str(i)]["pos"]
            neg = assignment[str(i)]["neg"]
            if not (forget_set & set(neg)):
                continue  # short-circuit: this survivor never saw a forgotten negative
            set_seed(cfg["seed"] + i)
            new_m, st = retrain_binary_model(i, pos, neg, forget_set,
                                             train_ds, cfg, device)
            ensemble.submodels[i] = new_m.to(device)
            shard_stats.append(st)

    elif variant == "slice_resume":
        # Sample-wise: retrain every sub-model whose pos OR neg overlaps the forget set.
        # When ovr_slices > 1, resume from the last clean slice checkpoint instead
        # of retraining from scratch (mirrors SISA's retrain_shard).
        num_slices = cfg.get("ovr_slices") or 1
        for i in range(c):
            pos = assignment[str(i)]["pos"]
            neg = assignment[str(i)]["neg"]
            if not (forget_set & (set(pos) | set(neg))):
                continue  # short-circuit: unaffected
            set_seed(cfg["seed"] + i)
            if num_slices > 1:
                class_dir = os.path.join(ovr_dir, f"class_{i}")
                new_m, st = resume_binary_from_slice(
                    i, pos, neg, forget_set, class_dir, train_ds, cfg, device)
            else:
                new_m, st = retrain_binary_model(
                    i, pos, neg, forget_set, train_ds, cfg, device)
            ensemble.submodels[i] = new_m.to(device)
            shard_stats.append(st)
        print(f"  retrained {len(shard_stats)} affected sub-model(s)")

    unlearn_time = time.time() - unlearn_start
    print(f"\n  Unlearning completed in {unlearn_time:.1f}s")

    print("\nEvaluating updated ensemble...")
    after = evaluate_snapshot(ensemble, c, device, test_loader,
                              retain_loader, forget_loader, cfg["seed"],
                              mia_test_loader=mia_test_loader)

    ovr_specific = {}
    if strategy == "class":
        forgotten = sorted({int(targets[i]) for i in forget_indices})
        ovr_specific.update(negative_footprint(
            assignment, forgotten, ensemble.active_classes))

    print(f"\n{'='*68}")
    print(f"{'Metric':<22} {'Before':>12}  {'After':>12}  {'Δ':>7}")
    print(f"{'='*68}")
    for name, b, a in [("Retain Accuracy", before["retain_acc"], after["retain_acc"]),
                       ("Forget Accuracy", before["forget_acc"], after["forget_acc"]),
                       ("Test Accuracy",   before["test_acc"],   after["test_acc"])]:
        d = a - b
        print(f"{name:<22} {b:>11.2f}%  {a:>11.2f}%  {('+' if d>=0 else ''):>0}{d:>6.2f}%")
    print("-" * 68)
    for name, b, a in [("MIA_L (loss)",    before["mia_l"], after["mia_l"]),
                       ("MIA_E (entropy)", before["mia_e"], after["mia_e"])]:
        d = a - b
        print(f"{name:<22} {b*100:>11.2f}%  {a*100:>11.2f}%  "
              f"{('+' if d>=0 else '')}{d*100:>6.2f}%   (ideal: 50%)")
    print("=" * 68)

    ovr_train_time = 0.0
    meta_path = os.path.join(ovr_dir, "ensemble_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            ovr_train_time = json.load(f).get("total_time_s", 0.0)

    _write_results(results_path, {
        "status": "complete", "dataset": dataset, "seed": cfg["seed"],
        "method": f"ovr_{variant}", "ovr_variant": variant,
        "forget_strategy": strategy,
        "forget_fraction": cfg.get("forget_fraction") if strategy == "random" else None,
        "forget_class": cfg.get("forget_class") if strategy == "class" else None,
        "forget_size": len(forget_indices), "retain_size": len(retain_indices),
        "neg_ratio": cfg["ovr_neg_ratio"], "ovr_slices": cfg["ovr_slices"],
        "unlearn_time_s": unlearn_time, "ovr_train_time_s": ovr_train_time,
        "before": before, "after": after,
        "ovr_specific": ovr_specific, "shard_stats": shard_stats,
    })
    print(f"\n  Results saved → {results_path}")


# entry point

def main():
    args = parse_args()
    cfg = merge(load_config(args.config), args)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        run_train(cfg, device)
    else:
        run_unlearn(cfg, device)


if __name__ == "__main__":
    main()
