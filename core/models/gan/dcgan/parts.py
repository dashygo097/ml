from typing import OrderedDict

import torch
from torch import nn

from .config import DCGANConfig


class DCGANGenerator(nn.Module):
    def __init__(self, config: DCGANConfig) -> None:
        super().__init__()
        self.config = config

        module_list = []
        self.init_size = (
            self.config.img_shape[0] // 2**config.n_gen_layers,
            self.config.img_shape[1] // 2**config.n_gen_layers,
        )

        self.linear_init = nn.Linear(
            self.config.latent_dim,
            self.config.gen_hidden_channels * self.init_size[0] * self.init_size[1],
        )
        self.bn2d_init = nn.BatchNorm1d(
            self.config.gen_hidden_channels * self.init_size[0] * self.init_size[1]
        )

        in_channels = self.config.gen_hidden_channels
        out_channels = self.config.gen_hidden_channels
        for index in range(config.n_gen_layers):
            module_list.extend(
                [
                    (
                        "conv2d_" + str(index),
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=self.config.gen_kernel_size,
                            stride=1,
                            padding=(self.config.gen_kernel_size - 1) // 2,
                        ),
                    ),
                    ("pixel_shuffle_" + str(index), nn.PixelShuffle(2)),
                    (
                        "bn2d_" + str(index),
                        nn.BatchNorm2d(out_channels),
                    ),
                    ("silu_" + str(index), nn.SiLU(inplace=True)),
                    ("dropout_" + str(index + 1), nn.Dropout(self.config.gen_dropout)),
                ],
            )
            in_channels = out_channels
            out_channels = out_channels // 2

        fit_channels = self.config.gen_hidden_channels // (2**config.n_gen_layers) * 2
        module_list.extend(
            [
                (
                    "conv2d_out",
                    nn.Conv2d(
                        fit_channels,
                        self.config.n_channels,
                        kernel_size=self.config.gen_kernel_size,
                        stride=1,
                        padding=(self.config.gen_kernel_size - 1) // 2,
                    ),
                ),
                ("act", nn.Tanh()),
            ]
        )

        self.seq = nn.Sequential(OrderedDict(module_list))
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L_dim = z.shape
        z = self.bn2d_init(self.linear_init(z))
        z = z.view(
            B,
            self.config.gen_hidden_channels,
            self.init_size[0],
            self.init_size[1],
        )
        z = self.seq(z)

        return z


class DCGANDiscriminator(nn.Module):
    def __init__(self, config: DCGANConfig) -> None:
        super().__init__()
        self.config = config
        module_list = []
        module_list.extend(
            [
                (
                    "conv2d_init",
                    nn.Conv2d(
                        self.config.n_channels,
                        self.config.dis_hidden_channels,
                        kernel_size=self.config.dis_kernel_size,
                        stride=2,
                        padding=(self.config.dis_kernel_size - 1) // 2,
                    ),
                ),
                ("silu_init", nn.SiLU(inplace=True)),
                ("dropout_init", nn.Dropout2d(self.config.dis_dropout)),
            ]
        )
        in_channels = self.config.dis_hidden_channels
        out_channels = self.config.dis_hidden_channels * 2
        img_shape = ((self.config.img_shape[0] + 1) // 2, self.config.img_shape[1] // 2)

        for index in range(config.n_dis_layers):
            module_list.extend(
                [
                    (
                        "conv2d_" + str(index + 1),
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=self.config.dis_kernel_size,
                            stride=2,
                            padding=(self.config.dis_kernel_size - 1) // 2,
                        ),
                    ),
                    ("silu_" + str(index + 1), nn.SiLU(inplace=True)),
                    (
                        "dropout_" + str(index + 1),
                        nn.Dropout2d(self.config.dis_dropout),
                    ),
                    ("bn2d_" + str(index), nn.BatchNorm2d(out_channels)),
                ]
            )
            in_channels = out_channels
            out_channels = out_channels * 2
            img_shape = (
                (img_shape[0] + 1) // 2,
                (img_shape[1] + 1) // 2,
            )

        fit_dim = out_channels // 2

        module_list.extend(
            [
                ("adpool2d", nn.AdaptiveAvgPool2d((1, 1))),
                ("flatten", nn.Flatten()),
                ("dense", nn.Linear(fit_dim, 1)),
                ("act", nn.Sigmoid()),
            ]
        )
        self.seq = nn.Sequential(OrderedDict(module_list))
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x).squeeze(-1)
