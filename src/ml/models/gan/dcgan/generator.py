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

        module_list.extend(
            [
                (
                    "conv2d_trans_init",
                    nn.ConvTranspose2d(
                        self.config.latent_dim,
                        self.config.gen_hidden_channels,
                        kernel_size=self.config.gen_kernel_size,
                        stride=2,
                        padding=0,
                        bias=False,
                    ),
                ),
                (
                    "bn2d_init",
                    nn.BatchNorm2d(self.config.gen_hidden_channels),
                ),
                (
                    "silu_init",
                    nn.SiLU(inplace=True),
                ),
            ]
        )

        in_channels = self.config.gen_hidden_channels
        out_channels = self.config.gen_hidden_channels
        for index in range(config.n_gen_layers):
            module_list.extend(
                [
                    (
                        "conv2d_trans_" + str(index),
                        nn.ConvTranspose2d(
                            in_channels,
                            out_channels,
                            kernel_size=self.config.gen_kernel_size,
                            stride=2,
                            padding=(self.config.gen_kernel_size - 1) // 2,
                            bias=False,
                        ),
                    ),
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
                    "conv2d_trans_out",
                    nn.ConvTranspose2d(
                        fit_channels,
                        self.config.n_channels,
                        kernel_size=self.config.gen_kernel_size,
                        stride=2,
                        padding=(self.config.gen_kernel_size - 1) // 2,
                        bias=False,
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

        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L_dim = z.shape
        z = z.view(B, L_dim).unsqueeze(-1).unsqueeze(-1)
        z = self.seq(z)

        return z
