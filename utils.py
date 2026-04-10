"""
Shared utilities for Machine Unlearning experiments.

Contents:
    - STATS           : normalisation constants for CIFAR-10 / CIFAR-100
    - set_seed()      : reproducibility helper
    - get_datasets()  : loads CIFAR train/test datasets with standard transforms
    - evaluate()      : loss + accuracy on a DataLoader
    - per_class_accuracy() : per-class accuracy tensor
    - save_checkpoint / load_checkpoint : checkpoint I/O
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import build_resnet18


# ── Normalisation statistics ──────────────────────────────────────────────────

STATS = {
    "CIFAR10":  {"mean": (0.4914, 0.4822, 0.4465),
                 "std":  (0.2023, 0.1994, 0.2010)},
    "CIFAR100": {"mean": (0.5071, 0.4867, 0.4408),
                 "std":  (0.2675, 0.2565, 0.2761)},
}


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Dataset helpers ───────────────────────────────────────────────────────────

def get_datasets(dataset: str,
                 data_root: str = "./data",
                 download: bool = True):
    """
    Load CIFAR-10 or CIFAR-100 with standard augmentation.

    Parameters
    ----------
    dataset : str
        ``"CIFAR10"`` or ``"CIFAR100"``.
    data_root : str
        Directory to download / cache datasets.
    download : bool
        Whether to download the dataset if not present.

    Returns
    -------
    (train_dataset, test_dataset)
        ``torchvision.datasets`` with the appropriate transforms applied.
    """
    mean, std = STATS[dataset]["mean"], STATS[dataset]["std"]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    DatasetClass = (torchvision.datasets.CIFAR10 if dataset == "CIFAR10"
                    else torchvision.datasets.CIFAR100)

    train_ds = DatasetClass(root=data_root, train=True,  download=download,
                            transform=train_transform)
    test_ds  = DatasetClass(root=data_root, train=False, download=download,
                            transform=test_transform)

    return train_ds, test_ds


def get_test_transform(dataset: str):
    """Return the deterministic (no augmentation) transform for evaluation."""
    mean, std = STATS[dataset]["mean"], STATS[dataset]["std"]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """
    Evaluate model on a data loader.

    Returns
    -------
    (avg_loss, accuracy_percent) : tuple[float, float]
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def per_class_accuracy(model: nn.Module,
                       loader: DataLoader,
                       num_classes: int,
                       device: torch.device) -> torch.Tensor:
    """
    Compute per-class accuracy over a data loader.

    Returns
    -------
    torch.Tensor of shape (num_classes,) with accuracy % per class.
    """
    model.eval()
    class_correct = torch.zeros(num_classes)
    class_total   = torch.zeros(num_classes)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)

        for c in range(num_classes):
            mask = labels == c
            class_correct[c] += preds[mask].eq(labels[mask]).sum().item()
            class_total[c]   += mask.sum().item()

    return 100.0 * class_correct / class_total.clamp(min=1)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str,
                    device: torch.device) -> nn.Module:
    """
    Load a saved checkpoint and return the model ready for use.

    The checkpoint must have been saved with the structure produced by
    the training script (keys: model_state, num_classes, epoch, etc.).

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pth file.
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    nn.Module  — model in eval() mode with weights loaded.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = build_resnet18(ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded  : {checkpoint_path}")
    print(f"Dataset : {ckpt['dataset']}  |  "
          f"Epoch: {ckpt['epoch']}  |  "
          f"Test acc: {ckpt['test_acc']:.2f}%")
    return model


def save_checkpoint(model: nn.Module,
                    path: str,
                    epoch: int,
                    test_acc: float,
                    dataset: str,
                    num_classes: int,
                    extra: dict = None) -> None:
    """
    Save a model checkpoint.

    Parameters
    ----------
    extra : dict, optional
        Any additional metadata to store (e.g. forget_set_size,
        unlearning_method, etc.).
    """
    payload = {
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "test_acc":    test_acc,
        "dataset":     dataset,
        "num_classes": num_classes,
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    print(f"Saved checkpoint → {path}")
