from typing import List, OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .config import HiFiGANConfig


def get_padding(kernel_size: int, dilation: int):
    return (kernel_size - 1) // 2 * dilation


class ResUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: List[int] = [1, 3, 5],
    ) -> None:
        super().__init__()

        module_list = []
        for i, d in enumerate(dilation):
            pad = get_padding(kernel_size, d)
            module_list.extend(
                [
                    (
                        f"conv1d_{i}",
                        weight_norm(
                            nn.Conv1d(
                                in_channels,
                                out_channels,
                                kernel_size,
                                padding=pad,
                                dilation=d,
                            ),
                        ),
                    ),
                    (f"leaky_relu_{i}", nn.LeakyReLU(0.1)),
                ]
            )
        self.seq = nn.Sequential(OrderedDict(module_list))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.seq(z)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: List[int] = [3, 7, 11],
        dilation: List[List[int]] = [[1, 3, 5]] * 3,
    ) -> None:
        super().__init__()
        self.units = []
        for _, (k, d) in enumerate(zip(kernels, dilation)):
            self.units.extend(
                [
                    ResUnit(in_channels, out_channels, k, d),
                ]
            )

        self.units = nn.ModuleList(self.units)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        res_outs = [unit(z) for unit in self.units]
        return sum(res_outs) / 3  # pyright: ignore


class MSDBlock(nn.Module):
    def __init__(self, config: HiFiGANConfig, use_spec_norm: bool = False) -> None:
        super().__init__()
        self.config = config
        num_layers = len(config.dis_msd_kernels)
        module_list = []
        norm_f = weight_norm if use_spec_norm else nn.utils.spectral_norm
        module_list.extend(
            [
                (
                    "conv1d_pre",
                    norm_f(
                        nn.Conv1d(
                            1,
                            config.dis_hidden_dim,
                            kernel_size=15,
                            stride=1,
                            padding=get_padding(15, 1),
                        ),
                    ),
                ),
            ]
        )
        for i in range(num_layers):
            module_list.extend(
                [
                    (
                        f"conv1d_{i}",
                        norm_f(
                            nn.Conv1d(
                                config.dis_hidden_dim * 2**i,
                                config.dis_hidden_dim * 2 ** (i + 1),
                                kernel_size=config.dis_msd_kernels[i],
                                stride=config.dis_msd_strides[i],
                                groups=config.dis_msd_groups[i],
                                padding=get_padding(config.dis_msd_kernels[i], 1),
                            ),
                        ),
                    ),
                    (f"leaky_relu_{i}", nn.LeakyReLU(0.1)),
                ]
            )

        module_list.extend(
            [
                (
                    "conv1d_post_pre",
                    norm_f(
                        nn.Conv1d(
                            config.dis_hidden_dim * 2**num_layers,
                            config.dis_hidden_dim * 2**num_layers,
                            kernel_size=config.dis_msd_kernels[-1],
                            stride=1,
                            groups=config.dis_msd_groups[-1],
                            padding=get_padding(config.dis_msd_kernels[-1], 1),
                        )
                    ),
                ),
                (
                    "conv1d_post",
                    norm_f(
                        nn.Conv1d(
                            config.dis_hidden_dim * 2**num_layers,
                            1,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                ),
            ]
        )
        self.seq = nn.Sequential(OrderedDict(module_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        return x


class MSD(nn.Module):
    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        self.config = config
        msd_blocks = []
        msd_blocks.append(MSDBlock(config, use_spec_norm=True))
        msd_blocks.extend(
            [MSDBlock(config) for _ in range(config.dis_msd_num_blocks - 1)]
        )

        self.msd_blocks = nn.ModuleList(msd_blocks)

        self.poolings = nn.ModuleList(
            [
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
                for _ in range(config.dis_msd_num_blocks - 1)
            ]
        )

    def forward(self, y: torch.Tensor) -> List[torch.Tensor]:
        y_outs = []
        for i, block in enumerate(self.msd_blocks):
            if i != 0:
                y = self.poolings[i - 1](y)
            y = block(y)
            y_outs.append(y.squeeze(1))

        return y_outs


class MPDBlock(nn.Module):
    def __init__(self, period: int, config: HiFiGANConfig) -> None:
        super().__init__()
        self.config = config
        self.period = period
        num_layers = len(config.dis_mpd_kernels)
        module_list = []
        module_list.extend(
            [
                (
                    "conv2d_pre",
                    weight_norm(
                        nn.Conv2d(
                            1,
                            config.dis_hidden_dim,
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            padding=(get_padding(5, 3), 0),
                        ),
                    ),
                )
            ]
        )

        for i in range(num_layers):
            module_list.extend(
                [
                    (
                        f"conv2d_{i}",
                        weight_norm(
                            nn.Conv2d(
                                config.dis_hidden_dim * 2**i,
                                config.dis_hidden_dim * 2 ** (i + 1),
                                kernel_size=(config.dis_mpd_kernels[i], 1),
                                stride=(3, 1),
                                padding=(get_padding(config.dis_mpd_kernels[i], 3), 0),
                            ),
                        ),
                    ),
                    (f"leaky_relu_{i}", nn.LeakyReLU(0.1)),
                ]
            )

        module_list.extend(
            [
                (
                    "conv2d_post_pre",
                    weight_norm(
                        nn.Conv2d(
                            config.dis_hidden_dim * 2**num_layers,
                            config.dis_hidden_dim * 2**num_layers,
                            kernel_size=(config.dis_mpd_kernels[-1], 1),
                            stride=(3, 1),
                            padding=(get_padding(config.dis_mpd_kernels[-1], 3), 0),
                        ),
                    ),
                ),
                (
                    "conv2d_post",
                    weight_norm(
                        nn.Conv2d(
                            config.dis_hidden_dim * 2**num_layers,
                            1,
                            kernel_size=(3, 1),
                            stride=(1, 1),
                            padding=(1, 0),
                        ),
                    ),
                ),
            ]
        )

        self.seq = nn.Sequential(OrderedDict(module_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if T % self.period != 0:
            # Padding
            x = nn.functional.pad(x, (0, self.period - T % self.period), "reflect")

        x = x.view(B, C, -1, self.period)
        x = self.seq(x)

        return torch.flatten(x, 1, -1)


class MPD(nn.Module):
    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        self.config = config
        self.mpd_blocks = nn.ModuleList(
            [MPDBlock(p, config) for p in config.dis_mpd_periods]
        )

    def forward(self, y: torch.Tensor) -> List[torch.Tensor]:
        y_outs = []
        for block in self.mpd_blocks:
            y_outs.append(block(y))
        return y_outs
