import torch
import torch.nn as nn
import torchvision.models as models


STATS = {
    "CIFAR10":  {"mean": (0.4914, 0.4822, 0.4465),
                 "std":  (0.2023, 0.1994, 0.2010)},
    "CIFAR100": {"mean": (0.5071, 0.4867, 0.4408),
                 "std":  (0.2675, 0.2565, 0.2761)},
}

# MODEL
def build_resnet18(num_classes: int, cifar_head: bool = True) -> nn.Module:
    """
    Returns a ResNet-18 adapted for CIFAR (32x32) input.

    Parameters
    ----------
    num_classes : int
        10 for CIFAR-10, 100 for CIFAR-100.
    cifar_head : bool
        If True, replaces the 7x7/stride-2 stem with a 3x3/stride-1 conv
        and removes the initial max-pool — standard for CIFAR benchmarks.

    Returns
    -------
    nn.Module
        ResNet-18 with the correct output head for the given dataset.
    """
    model = models.resnet18(weights=None)   # always train/load from scratch

    if cifar_head:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                padding=1, bias=False)
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# EVAL
@torch.no_grad()
def evaluate(model: nn.Module,
             loader: torch.utils.data.DataLoader,
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
                       loader: torch.utils.data.DataLoader,
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
    the training notebook (keys: model_state, num_classes, epoch, etc.).

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
