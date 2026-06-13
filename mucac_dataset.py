"""
PyTorch Dataset for MUCAC (multi-label CelebA subset).

Labels: Male, Young, Smiling  — each is 0/1, returned as float32 tensor of shape (3,).
Loss:   BCEWithLogitsLoss (do NOT apply sigmoid in the model).
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T


LABEL_COLS = ["Male", "Young", "Smiling"]

# ImageNet-style stats work well for 128×128 CelebA crops
MUCAC_MEAN = (0.5063, 0.4258, 0.3832)
MUCAC_STD  = (0.3107, 0.2904, 0.2897)


def _train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(MUCAC_MEAN, MUCAC_STD),
    ])


def _eval_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(MUCAC_MEAN, MUCAC_STD),
    ])


class MuCACDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to train.csv or test.csv produced by build_mucac_dataset.py.
    img_dir : str
        Directory containing the images referenced by the CSV (train/ or test/).
    transform : callable, optional
        Transform applied to each PIL image.
    forget_identity : int or None
        If set, samples with this identity are flagged via ``is_forget``.
    """

    def __init__(self, csv_path: str, img_dir: str,
                 transform=None, forget_identity: int | None = None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform

        self.labels = torch.tensor(
            self.df[LABEL_COLS].values, dtype=torch.float32
        )

        if forget_identity is not None:
            self.forget_mask = (self.df["identity"] == forget_identity).values
        else:
            self.forget_mask = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["filename"])
        img  = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

    def forget_indices(self, identity: int) -> list[int]:
        """Indices belonging to the given identity (D_f)."""
        return self.df.index[self.df["identity"] == identity].tolist()

    def retain_indices(self, identity: int) -> list[int]:
        """Indices NOT belonging to the given identity (D_r)."""
        return self.df.index[self.df["identity"] != identity].tolist()


# loaders

def get_mucac_loaders(data_root: str,
                      batch_size: int = 64,
                      num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) with standard augmentation.

    Parameters
    ----------
    data_root : str
        Directory containing mucac_dataset/ (i.e. the repo data/ folder).
    """
    base = os.path.join(data_root, "mucac_dataset")

    train_ds = MuCACDataset(
        csv_path  = os.path.join(base, "train.csv"),
        img_dir   = os.path.join(base, "train"),
        transform = _train_transform(),
    )
    test_ds = MuCACDataset(
        csv_path  = os.path.join(base, "test.csv"),
        img_dir   = os.path.join(base, "test"),
        transform = _eval_transform(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def get_forget_retain_loaders(train_ds: MuCACDataset,
                               forget_identity: int,
                               batch_size: int = 64,
                               num_workers: int = 4
                               ) -> tuple[DataLoader, DataLoader]:
    """
    Split train_ds into forget / retain loaders for a given identity.
    Both use the eval transform (no augmentation) — pass a train_ds built
    with _eval_transform() when you need deterministic behaviour for unlearning.
    """
    f_idx = train_ds.forget_indices(forget_identity)
    r_idx = train_ds.retain_indices(forget_identity)

    forget_loader = DataLoader(Subset(train_ds, f_idx), batch_size=batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True)
    retain_loader = DataLoader(Subset(train_ds, r_idx), batch_size=batch_size,
                               shuffle=True,  num_workers=num_workers,
                               pin_memory=True)

    return forget_loader, retain_loader


# sisa sharding

def identity_shard(df, num_shards: int, seed: int) -> list[list[int]]:
    """
    Partition dataset indices into num_shards disjoint shards such that all
    images of a given identity land in the same shard.

    This guarantees that any identity in D_f is contained entirely within
    one shard — the same guarantee that enables efficient SISA unlearning.

    Returns list of num_shards lists of integer DataFrame row indices.
    """
    rng        = np.random.RandomState(seed)
    identities = df["identity"].unique().copy()
    rng.shuffle(identities)

    shard_for_id = {int(id_): i % num_shards for i, id_ in enumerate(identities)}

    shards: list[list[int]] = [[] for _ in range(num_shards)]
    for idx in df.index:
        shards[shard_for_id[int(df.loc[idx, "identity"])]].append(int(idx))

    return shards


@torch.no_grad()
# ensemble evaluation

def ensemble_evaluate_multilabel(models: list, loader: DataLoader,
                                  device: torch.device) -> dict[str, float]:
    """
    Per-label metrics for a SISA ensemble.
    Aggregation: mean sigmoid probability across all shard models, threshold 0.5.
    """
    import torch.nn as nn
    for m in models:
        m.eval()

    all_preds  = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        avg_prob = torch.stack(
            [m(imgs).sigmoid() for m in models]
        ).mean(dim=0).cpu()                      # (B, L)
        preds = (avg_prob > 0.5).float()
        all_preds.append(preds)
        all_labels.append(labels)

    preds  = torch.cat(all_preds,  dim=0)
    labels = torch.cat(all_labels, dim=0)

    result = {}
    f1s, bal_accs, accs = [], [], []

    for i, col in enumerate(LABEL_COLS):
        p = preds[:, i];  y = labels[:, i]
        tp = ((p == 1) & (y == 1)).sum().float()
        fp = ((p == 1) & (y == 0)).sum().float()
        fn = ((p == 0) & (y == 1)).sum().float()
        tn = ((p == 0) & (y == 0)).sum().float()

        precision   = tp / (tp + fp).clamp(min=1)
        recall      = tp / (tp + fn).clamp(min=1)
        specificity = tn / (tn + fp).clamp(min=1)

        f1      = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        bal_acc = (recall + specificity) / 2
        acc     = (tp + tn) / len(y)

        result[f"{col}_acc"]     = float(acc * 100)
        result[f"{col}_f1"]      = float(f1 * 100)
        result[f"{col}_bal_acc"] = float(bal_acc * 100)
        accs.append(float(acc * 100))
        f1s.append(float(f1 * 100))
        bal_accs.append(float(bal_acc * 100))

    result["mean_acc"]     = sum(accs)     / len(accs)
    result["mean_f1"]      = sum(f1s)      / len(f1s)
    result["mean_bal_acc"] = sum(bal_accs) / len(bal_accs)
    return result


@torch.no_grad()
# per-label evaluation

def evaluate_multilabel(model, loader, device) -> dict[str, float]:
    """
    Per-label metrics: accuracy, F1, balanced accuracy.

    Returns a flat dict, e.g.:
        Male_acc, Male_f1, Male_bal_acc,
        Young_acc, Young_f1, Young_bal_acc,
        Smiling_acc, Smiling_f1, Smiling_bal_acc,
        mean_acc, mean_f1, mean_bal_acc

    Threshold: logit > 0  (equivalent to sigmoid(logit) > 0.5).
    Use F1 and bal_acc for imbalanced labels (especially Young ~77% pos).
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        logits = model(imgs)
        preds  = (logits > 0).float()          # threshold on raw logit
        all_preds.append(preds.cpu())
        all_labels.append(labels)

    preds  = torch.cat(all_preds,  dim=0)      # (N, 3)
    labels = torch.cat(all_labels, dim=0)      # (N, 3)

    result = {}
    f1s, bal_accs, accs = [], [], []

    for i, col in enumerate(LABEL_COLS):
        p = preds[:, i]
        y = labels[:, i]

        tp = ((p == 1) & (y == 1)).sum().float()
        fp = ((p == 1) & (y == 0)).sum().float()
        fn = ((p == 0) & (y == 1)).sum().float()
        tn = ((p == 0) & (y == 0)).sum().float()

        precision   = tp / (tp + fp).clamp(min=1)
        recall      = tp / (tp + fn).clamp(min=1)
        specificity = tn / (tn + fp).clamp(min=1)

        f1      = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        bal_acc = (recall + specificity) / 2
        acc     = (tp + tn) / len(y)

        result[f"{col}_acc"]     = float(acc * 100)
        result[f"{col}_f1"]      = float(f1 * 100)
        result[f"{col}_bal_acc"] = float(bal_acc * 100)

        accs.append(float(acc * 100))
        f1s.append(float(f1 * 100))
        bal_accs.append(float(bal_acc * 100))

    result["mean_acc"]     = sum(accs)     / len(accs)
    result["mean_f1"]      = sum(f1s)      / len(f1s)
    result["mean_bal_acc"] = sum(bal_accs) / len(bal_accs)
    return result
