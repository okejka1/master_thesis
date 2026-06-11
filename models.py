"""
Model definitions for the Machine Unlearning experiments.

Currently supports:
    - ResNet-18 adapted for CIFAR (32×32) inputs.
    - ResNet-18 multi-label head for MUCAC (128×128 CelebA crops).

"""

import torch.nn as nn
import torchvision.models as models


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


def build_resnet18_multilabel(num_labels: int = 3) -> nn.Module:
    """
    ResNet-18 for multi-label classification (MUCAC: Male/Young/Smiling).

    Input : 128×128 RGB images (standard ImageNet stem — no CIFAR hack).
    Output: raw logits of shape (B, num_labels).
    Loss  : BCEWithLogitsLoss — do NOT apply sigmoid inside the model.

    Parameters
    ----------
    num_labels : int
        Number of binary labels (default 3 for MUCAC).
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model
