"""
Model definitions for the Machine Unlearning experiments.

Currently supports:
    - ResNet-18 adapted for CIFAR (32×32) inputs.

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
