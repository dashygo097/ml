from typing import Optional

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        base_width: int = 64,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * groups

        self.conv2d_0 = nn.Conv2d(
            in_channels,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.bn2d_0 = nn.BatchNorm2d(width)
        self.relu_0 = nn.ReLU(inplace=True)
        self.conv2d_1 = nn.Conv2d(
            width,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2d_1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id = x.clone()
        if self.downsample is not None:
            x_id = self.downsample(x)

        x = self.relu_0(self.bn2d_0(self.conv2d_0(x)))
        x = self.bn2d_1(self.conv2d_1(x))

        x += x_id
        x = self.relu_1(x)
        return x


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        base_width: int = 64,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * groups

        self.conv2d_0 = nn.Conv2d(
            in_channels,
            width,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2d_0 = nn.BatchNorm2d(width)
        self.relu_0 = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn2d_1 = nn.BatchNorm2d(width)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(
            width,
            width * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2d_2 = nn.BatchNorm2d(width * 4)
        self.relu_2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id = x.clone()
        if self.downsample is not None:
            x_id = self.downsample(x)

        x = self.relu_0(self.bn2d_0(self.conv2d_0(x)))
        x = self.relu_1(self.bn2d_1(self.conv2d_1(x)))
        x = self.bn2d_2(self.conv2d_2(x))

        x += x_id
        x = self.relu_2(x)
        return x
