"""
mia.py — Membership Inference Attack (MIA) evaluation for Machine Unlearning.

Two attack variants are implemented, following Trippa et al. (2024) / ∇τ paper,
which in turn follows Kurmanji et al. (SCRUB, 2023) and Foster et al. (SSD, 2023):

    MIA_L  — Loss-based attack:    attacker feature = per-sample cross-entropy loss
    MIA_E  — Entropy-based attack: attacker feature = Shannon entropy of softmax output

Protocol
--------
1. Compute scalar features for forget set (label=1) and test set (label=0).
2. Subsample the test set to balance class sizes.
3. Train a Logistic Regression binary classifier with 5-fold stratified CV.
4. Report mean CV accuracy.

Ideal result  : 50% accuracy — model treats Df identically to unseen test data.
Original model: typically 60–90% — model memorised Df, attacker can distinguish.

Note on the Streisand effect (Golatkar et al., 2020):
    A score of 0% is *not* ideal — it signals over-forgetting, which itself leaks
    information. The target is 50% (indistinguishable from test set).

References
----------
- Trippa et al.  (2024) ∇τ  https://arxiv.org/abs/2403.14339
- Kurmanji et al.(2023) SCRUB — Towards Unbounded Machine Unlearning
- Foster et al.  (2023) SSD  — Fast Machine Unlearning Without Retraining
- Golatkar et al.(2020) Eternal Sunshine of the Spotless Net
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
) -> np.ndarray:
    """
    Compute a scalar feature for every sample in `loader`.

    Parameters
    ----------
    model   : nn.Module in eval mode
    loader  : DataLoader (no shuffle required)
    device  : torch.device
    method  : ``"loss"``    — per-sample cross-entropy loss
              ``"entropy"`` — Shannon entropy of softmax output

    Returns
    -------
    np.ndarray of shape (N,) — one scalar per sample.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    features = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        if method == "loss":
            vals = criterion(logits, labels)                          # (B,)
        elif method == "entropy":
            probs = F.softmax(logits, dim=1)
            vals  = -(probs * torch.log(probs + 1e-12)).sum(dim=1)   # (B,)
        else:
            raise ValueError(f"Unknown MIA method: {method!r}. "
                             "Choose 'loss' or 'entropy'.")

        features.append(vals.cpu().numpy())

    return np.concatenate(features)


@torch.no_grad()
def _compute_features_ensemble(
    models: list,
    loader: DataLoader,
    device: torch.device,
    aggregation: str,
    method: str,
) -> np.ndarray:
    """
    Compute scalar features for a *SISA ensemble* of models.

    Softmax probabilities are averaged across all shard models (soft vote),
    then loss / entropy is computed from the averaged distribution.

    Parameters
    ----------
    models      : list[nn.Module]
    loader      : DataLoader
    device      : torch.device
    aggregation : ``"soft_vote"`` | ``"majority_vote"`` (both use soft-avg probs)
    method      : ``"loss"`` | ``"entropy"``

    Returns
    -------
    np.ndarray of shape (N,)
    """
    for m in models:
        m.eval()

    features = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Average softmax probabilities across all shards
        avg_probs = None
        for m in models:
            probs     = F.softmax(m(images), dim=1)
            avg_probs = probs if avg_probs is None else avg_probs + probs
        avg_probs = avg_probs / len(models)

        if method == "loss":
            log_probs = torch.log(avg_probs + 1e-12)
            vals = F.nll_loss(log_probs, labels, reduction="none")   # (B,)
        elif method == "entropy":
            vals = -(avg_probs * torch.log(avg_probs + 1e-12)).sum(dim=1)
        else:
            raise ValueError(f"Unknown MIA method: {method!r}.")

        features.append(vals.cpu().numpy())

    return np.concatenate(features)


# ── Binary attacker ───────────────────────────────────────────────────────────

def _train_attacker(
    forget_features: np.ndarray,
    test_features: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """
    Train a Logistic Regression MIA attacker with stratified k-fold CV.

    The forget set samples receive label 1 (member) and test set samples
    receive label 0 (non-member). The test set is subsampled to match the
    forget set size so that both classes are balanced.

    Parameters
    ----------
    forget_features : (N_f,) scalar features for the forget set
    test_features   : (N_t,) scalar features for the test set
    n_splits        : number of CV folds (default 5, as per user spec)
    seed            : random seed for reproducibility

    Returns
    -------
    float — mean CV accuracy (0–1).
            Ideal = 0.50 (random guessing = model forgot Df perfectly).
    """
    rng = np.random.RandomState(seed)

    # ── Balance classes: subsample test to match forget size ──────────────────
    n_forget = len(forget_features)
    if len(test_features) > n_forget:
        idx = rng.choice(len(test_features), size=n_forget, replace=False)
        test_features = test_features[idx]

    X = np.concatenate([forget_features, test_features]).reshape(-1, 1)
    y = np.concatenate([
        np.ones(len(forget_features), dtype=int),
        np.zeros(len(test_features),  dtype=int),
    ])

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    return float(scores.mean())


# ── Public API — single model ─────────────────────────────────────────────────

def mia_attack(
    model: nn.Module,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    method: str = "loss",
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """
    Run a Membership Inference Attack against a single model.

    Parameters
    ----------
    model         : nn.Module
    forget_loader : DataLoader for the forget set D_f
    test_loader   : DataLoader for the held-out test set D_test
    device        : torch.device
    method        : ``"loss"``    → MIA_L (Kurmanji et al.)
                    ``"entropy"`` → MIA_E (Foster et al.)
    n_splits      : CV folds for the logistic regression attacker
    seed          : random seed

    Returns
    -------
    float — mean CV accuracy (0–1). Closest to 0.50 = best unlearning.
    """
    forget_feats = _compute_features(model, forget_loader, device, method)
    test_feats   = _compute_features(model, test_loader,   device, method)
    return _train_attacker(forget_feats, test_feats, n_splits=n_splits, seed=seed)


def run_mia_suite(
    model: nn.Module,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    label: str = "",
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    Convenience wrapper: compute both MIA_L and MIA_E for a single model,
    print results with a 5-fold CV logistic regression attacker.

    Parameters
    ----------
    label : str  descriptive label for printed output (e.g. ``"Original"``)

    Returns
    -------
    dict with keys ``"mia_l"`` and ``"mia_e"`` (float, 0–1).
    """
    mia_l = mia_attack(model, forget_loader, test_loader, device,
                       method="loss",    n_splits=n_splits, seed=seed)
    mia_e = mia_attack(model, forget_loader, test_loader, device,
                       method="entropy", n_splits=n_splits, seed=seed)

    prefix = f"[{label}] " if label else ""
    _print_mia(prefix, mia_l, mia_e)
    return {"mia_l": mia_l, "mia_e": mia_e}


# ── Public API — SISA ensemble ────────────────────────────────────────────────

def mia_attack_ensemble(
    models: list,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    aggregation: str = "soft_vote",
    method: str = "loss",
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """
    Run a Membership Inference Attack against a SISA ensemble.

    Parameters
    ----------
    models        : list[nn.Module]  all shard models
    forget_loader : DataLoader
    test_loader   : DataLoader
    device        : torch.device
    aggregation   : ``"soft_vote"`` | ``"majority_vote"``
    method        : ``"loss"`` | ``"entropy"``
    n_splits      : CV folds
    seed          : random seed

    Returns
    -------
    float — mean CV accuracy (0–1), ideal 0.50.
    """
    forget_feats = _compute_features_ensemble(
        models, forget_loader, device, aggregation, method)
    test_feats   = _compute_features_ensemble(
        models, test_loader,   device, aggregation, method)
    return _train_attacker(forget_feats, test_feats, n_splits=n_splits, seed=seed)


def run_mia_suite_ensemble(
    models: list,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    aggregation: str = "soft_vote",
    label: str = "",
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    Convenience wrapper: compute both MIA_L and MIA_E for a SISA ensemble.

    Returns
    -------
    dict with keys ``"mia_l"`` and ``"mia_e"`` (float, 0–1).
    """
    mia_l = mia_attack_ensemble(models, forget_loader, test_loader, device,
                                aggregation=aggregation, method="loss",
                                n_splits=n_splits, seed=seed)
    mia_e = mia_attack_ensemble(models, forget_loader, test_loader, device,
                                aggregation=aggregation, method="entropy",
                                n_splits=n_splits, seed=seed)

    prefix = f"[{label}] " if label else ""
    _print_mia(prefix, mia_l, mia_e)
    return {"mia_l": mia_l, "mia_e": mia_e}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _print_mia(prefix: str, mia_l: float, mia_e: float) -> None:
    """Pretty-print MIA results with a quality indicator."""
    def _tag(v: float) -> str:
        return "✓ near-random (good)" if abs(v - 0.5) < 0.05 else ""

    print(f"  {prefix}MIA_L (loss-based)    : {mia_l * 100:.2f}%  {_tag(mia_l)}")
    print(f"  {prefix}MIA_E (entropy-based) : {mia_e * 100:.2f}%  {_tag(mia_e)}")
