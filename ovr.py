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

# Sample-wise unlearning (retrain affected sub-models):
    python ovr.py --mode unlearn --config configs/cifar10.yaml \
                  --forget-strategy random --forget-fraction 0.01 --ovr-variant slice_resume

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
from mia import _train_attacker, run_mia_suite
from utils import (
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


# ── Config helpers ────────────────────────────────────────────────────────────

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


# ── Binary sub-model ──────────────────────────────────────────────────────────

def build_ovr_binary() -> nn.Module:
    """ResNet-18 (CIFAR stem) with a single-logit head for binary one-vs-rest."""
    return build_resnet18(num_classes=1)


def load_binary(path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    m = build_ovr_binary().to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


# ── Positives / negatives per class ─────────────────────────────────────────

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


# ── Binary view dataset ───────────────────────────────────────────────────────

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


# ── One binary training run ───────────────────────────────────────────────────

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
    """Train one binary sub-model.  Persists to *save_path* if given (else the
    model is returned without being written to disk — used during retraining)."""
    epochs = ovr_epochs_of(cfg)
    model = build_ovr_binary().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg["lr"],
                          momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"], nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"])

    loader = balanced_loader(base_train, pos, neg, cfg["batch_size"])

    if verbose:
        print(f"  class {class_id:>3}  |  {len(pos)} pos / {len(neg)} neg  "
              f"|  {epochs} epochs")

    for epoch in range(1, epochs + 1):
        loss, acc = train_binary_epoch(model, loader, criterion, optimizer, device)
        scheduler.step()
        if verbose and (epoch == 1 or epoch % max(1, epochs // 5) == 0
                        or epoch == epochs):
            print(f"      epoch {epoch:>3}  lr {scheduler.get_last_lr()[0]:.5f}  "
                  f"bce {loss:.4f}  bal-acc {acc:.2f}%")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_checkpoint(
            model, save_path, epoch=epochs, test_acc=0.0,
            dataset=cfg["dataset"], num_classes=num_classes_of(cfg["dataset"]),
            extra={"class_id": class_id, "binary": True,
                   "n_pos": len(pos), "n_neg": len(neg),
                   "neg_ratio": cfg["ovr_neg_ratio"], "config": cfg},
        )
    return model


# ── The ensemble adapter ──────────────────────────────────────────────────────

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


# ── Forget set construction (identical semantics to unlearn_naive/sisa) ──────

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


# ── Sub-model retraining (drop_neg_retrain / slice_resume) ──────────────────

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


# ── OVR-specific metrics ──────────────────────────────────────────────────────

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


@torch.no_grad()
def negative_membership_mia(ensemble: OvREnsemble,
                            forget_loader: DataLoader,
                            testk_loader: DataLoader,
                            device: torch.device,
                            seed: int) -> float:
    """OVR-specific attack: after dropping f_k, can an attacker tell that the
    forgotten class-k TRAIN samples were used as NEGATIVES in surviving models?

    Feature = mean sigmoid over active sub-models (a sample suppressed as a
    negative reads lower).  Members (label 1) = forget train-class-k, non-members
    (label 0) = held-out test-class-k.  Score near 50% ⇒ footprint undetectable.
    Returns 0–1 accuracy, or -1.0 if there is nothing to attack."""
    active = ensemble.active.to(device)

    def feats(loader):
        out = []
        for images, _ in loader:
            images = images.to(device)
            logits = torch.stack([m(images).squeeze(1)
                                  for m in ensemble.submodels], dim=1)  # (B,c)
            probs = torch.sigmoid(logits)
            probs = probs.masked_fill(~active.unsqueeze(0), float("nan"))
            out.append(torch.nanmean(probs, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    f = feats(forget_loader)
    t = feats(testk_loader)
    if len(f) == 0 or len(t) == 0:
        return -1.0
    return _train_attacker(f, t, seed=seed)


# ── Eval bundle (one ensemble snapshot → all metrics) ───────────────────────

def evaluate_snapshot(ensemble, num_classes, device,
                      test_loader, retain_loader, forget_loader, seed):
    ce = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(ensemble, test_loader, ce, device)
    ret_loss, ret_acc = evaluate(ensemble, retain_loader, ce, device)
    fgt_loss, fgt_acc = evaluate(ensemble, forget_loader, ce, device)
    mia = run_mia_suite(ensemble, forget_loader, test_loader, device, seed=seed)
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


# ── Results writer ────────────────────────────────────────────────────────────

def _write_results(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════════════
#  MODE: train
# ════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════
#  MODE: unlearn
# ════════════════════════════════════════════════════════════════════════════

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

    # ── Data ──────────────────────────────────────────────────────────────────
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

    # ── Load original ensemble ──────────────────────────────────────────────
    print("Loading original OVR ensemble...")
    submodels = [load_binary(os.path.join(ovr_dir, f"class_{i}", "model.pth"),
                             device) for i in range(c)]
    ensemble = OvREnsemble(submodels, c).to(device)

    print("Evaluating original ensemble (this includes 2× MIA)...")
    before = evaluate_snapshot(ensemble, c, device, test_loader,
                               retain_loader, forget_loader, cfg["seed"])

    # ── Unlearn ──────────────────────────────────────────────────────────────
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
        for i in range(c):
            pos = assignment[str(i)]["pos"]
            neg = assignment[str(i)]["neg"]
            if not (forget_set & (set(pos) | set(neg))):
                continue  # short-circuit: unaffected
            set_seed(cfg["seed"] + i)
            new_m, st = retrain_binary_model(i, pos, neg, forget_set,
                                             train_ds, cfg, device)
            ensemble.submodels[i] = new_m.to(device)
            shard_stats.append(st)
        print(f"  retrained {len(shard_stats)} affected sub-model(s)")

    unlearn_time = time.time() - unlearn_start
    print(f"\n  Unlearning completed in {unlearn_time:.1f}s")

    # ── Evaluate updated ensemble ────────────────────────────────────────────
    print("\nEvaluating updated ensemble...")
    after = evaluate_snapshot(ensemble, c, device, test_loader,
                              retain_loader, forget_loader, cfg["seed"])

    # ── OVR-specific metrics ─────────────────────────────────────────────────
    ovr_specific = {}
    if strategy == "class":
        forgotten = sorted({int(targets[i]) for i in forget_indices})
        ovr_specific.update(negative_footprint(
            assignment, forgotten, ensemble.active_classes))
        # negative-membership MIA needs held-out test samples of the forgotten class
        test_targets = np.array(test_ds.targets)
        testk_idx = np.where(np.isin(test_targets, forgotten))[0].tolist()
        testk_loader = DataLoader(torch.utils.data.Subset(test_ds, testk_idx),
                                  batch_size=cfg["batch_size"], shuffle=False,
                                  num_workers=2, pin_memory=True)
        ovr_specific["neg_membership_mia"] = negative_membership_mia(
            ensemble, forget_loader, testk_loader, device, cfg["seed"])

    # ── Side-by-side table ───────────────────────────────────────────────────
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
    if "neg_membership_mia" in ovr_specific:
        nm = ovr_specific["neg_membership_mia"]
        print("-" * 68)
        if nm < 0:
            print("Neg-membership MIA     : n/a")
        else:
            print(f"Neg-membership MIA     : {nm*100:>11.2f}%        "
                  f"(ideal 50% ⇒ residual negative footprint undetectable)")
        print(f"Residual neg footprint : {ovr_specific['neg_footprint_frac']*100:.1f}% "
              f"of forgotten samples still negatives in "
              f"{ovr_specific['models_touched']}/{ovr_specific['n_active_models']} survivors")
    print("=" * 68)

    ovr_train_time = 0.0
    meta_path = os.path.join(ovr_dir, "ensemble_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            ovr_train_time = json.load(f).get("total_time_s", 0.0)

    _write_results(results_path, {
        "status": "complete", "dataset": dataset, "seed": cfg["seed"],
        "method": "ovr", "ovr_variant": variant,
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


# ── Entry point ───────────────────────────────────────────────────────────────

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
