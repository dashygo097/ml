from typing import Optional

import torch

from .backbone import BasicBlock, Bottleneck
from .model import ResNet


def resnet18(
    in_channels: int,
    n_classes: int,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    **kwargs,
):
    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
        n_classes=n_classes,
        **kwargs,
    )
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    return model


def resnet34(
    in_channels: int,
    n_classes: int,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    **kwargs,
):
    model = ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        n_classes=n_classes,
        **kwargs,
    )
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def resnet50(
    in_channels: int,
    n_classes: int,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    **kwargs,
):
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        n_classes=n_classes,
        **kwargs,
    )
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def resetnet101(
    in_channels: int,
    n_classes: int,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    **kwargs,
):
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        in_channels=in_channels,
        n_classes=n_classes,
        **kwargs,
    )
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def resnet152(
    in_channels: int,
    n_classes: int,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    **kwargs,
):
    model = ResNet(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        in_channels=in_channels,
        n_classes=n_classes,
        **kwargs,
    )
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model
