from typing import OrderedDict, Union

import torch
from torch import nn

from .backbone import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        base_width: int = 64,
        groups: int = 1,
        in_channels: int = 1,
    ):
        super().__init__()

        self.in_channels = base_width

        self.conv2d_0 = nn.Conv2d(
            in_channels,
            base_width,
            kernel_size=7,
            stride=2,
            padding=3,
            groups=groups,
            bias=False,
        )

        self.bn2d_0 = nn.BatchNorm2d(base_width)
        self.relu_0 = nn.ReLU(inplace=True)

        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        module_list = []
        module_list.extend(
            [
                (
                    f"lr_{i}",
                    self._make_layer(
                        block,
                        layers[i],
                        base_width * 2**i if i > 0 else base_width,
                        stride=2 if i > 0 else 1,
                        base_width=base_width,
                        groups=groups,
                    ),
                )
                for i in range(len(layers))
            ]
        )

        self.seq = nn.Sequential(OrderedDict(module_list))
        self.avgpool2d_0 = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        n_layers: int,
        out_channels: int,
        stride: int = 1,
        base_width: int = 64,
        groups: int = 1,
    ) -> nn.Sequential:
        in_channels = self.in_channels
        downsample = None
        layers = []

        if stride != 1 or (in_channels != out_channels * block.expansion):
            downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv2d_0",
                            nn.Conv2d(
                                in_channels,
                                out_channels * block.expansion,
                                kernel_size=1,
                                stride=stride,
                                bias=False,
                            ),
                        ),
                        ("bn2d_0", nn.BatchNorm2d(out_channels * block.expansion)),
                    ]
                )
            )

        layers.append(
            (
                "blk_0",
                block(
                    in_channels,
                    out_channels,
                    stride=stride,
                    downsample=downsample,
                ),
            )
        )

        self.in_channels = out_channels * block.expansion
        for i in range(1, n_layers):
            layers.append(
                (
                    f"blk_{i}",
                    block(
                        self.in_channels,
                        out_channels,
                        groups=groups,
                        base_width=base_width,
                    ),
                )
            )
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_0(self.bn2d_0(self.conv2d_0(x)))
        x = self.maxpool2d_0(x)
        x = self.seq(x)
        x = self.avgpool2d_0(x)
        return x
