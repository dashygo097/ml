from typing import OrderedDict

import torch
from torch import nn

from .config import GANConfig


class GAN(nn.Module):
    def __init__(self, config: GANConfig) -> None:
        super().__init__()
        self.config = config
        self.generator = GANGenerator(config)
        self.discriminator = GANDiscriminator(config)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def set_discriminator(self, discriminator: nn.Module) -> None:
        self.discriminator = discriminator


class GANGenerator(nn.Module):
    def __init__(self, config: GANConfig) -> None:
        super().__init__()
        self.config = config

        module_list = []
        module_list.extend(
            [
                (
                    "linear_0",
                    nn.Linear(self.config.latent_dim, self.config.gen_hidden_dim),
                ),
                ("leaky_relu_0", nn.LeakyReLU(0.1, inplace=True)),
                ("dropout_0", nn.Dropout(self.config.dis_dropout)),
            ],
        )

        for index in range(config.n_gen_layers):
            in_dim = self.config.gen_hidden_dim * (2**index)
            module_list.extend(
                [
                    ("linear_" + str(index + 1), nn.Linear(in_dim, in_dim * 2)),
                    (
                        "bn1d_" + str(index),
                        nn.BatchNorm1d(in_dim * 2, 0.8),
                    ),
                    (
                        "leaky_relu_" + str(index + 1),
                        nn.LeakyReLU(0.2, inplace=True),
                    ),
                    ("dropout_" + str(index + 1), nn.Dropout(self.config.dis_dropout)),
                ],
            )
        fit_dim = self.config.gen_hidden_dim * (2**config.n_gen_layers)
        module_list.extend(
            [
                ("dense", nn.Linear(fit_dim, config.io_dim)),
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L_dim = z.shape

        z = self.seq(z)
        z = z.view(
            B,
            self.config.n_channels,
            self.config.img_shape[0],
            self.config.img_shape[1],
        )
        return z


class GANDiscriminator(nn.Module):
    def __init__(self, config: GANConfig) -> None:
        super().__init__()

        self.config = config

        module_list = []
        module_list.extend(
            [
                (
                    "linear_0",
                    nn.Linear(
                        config.io_dim * config.n_channels, self.config.dis_hidden_dim
                    ),
                ),
                ("leaky_relu_0", nn.LeakyReLU(0.2, inplace=True)),
            ]
        )

        for index in range(config.n_dis_layers):
            in_dim = self.config.dis_hidden_dim // (2**index)
            module_list.extend(
                [
                    ("linear_" + str(index + 1), nn.Linear(in_dim, in_dim // 2)),
                    ("leaky_relu_" + str(index + 1), nn.LeakyReLU(0.1, inplace=True)),
                ]
            )

        fit_dim = self.config.dis_hidden_dim // (2**config.n_dis_layers)
        module_list.extend(
            [
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

    def forward(self, x):
        B, C, W, H = x.shape

        x = x.view(B, C * W * H)
        x = self.seq(x)
        return x.squeeze(-1)
