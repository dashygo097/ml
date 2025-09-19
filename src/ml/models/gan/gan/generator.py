from typing import OrderedDict

import torch
from torch import nn

from .config import GANGeneratorConfig


class GANGenerator(nn.Module):
    def __init__(self, config: GANGeneratorConfig) -> None:
        super().__init__()
        self.config = config

        module_list = []
        module_list.extend(
            [
                (
                    "linear_0",
                    nn.Linear(self.config.latent_dim, self.config.hidden_dim),
                ),
                ("leaky_relu_0", nn.LeakyReLU(0.2, inplace=True)),
                ("dropout_0", nn.Dropout(self.config.dropout)),
            ],
        )

        for index in range(config.n_layers):
            in_dim = self.config.hidden_dim * (2**index)
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
                    ("dropout_" + str(index + 1), nn.Dropout(self.config.dropout)),
                ],
            )
        fit_dim = self.config.hidden_dim * (2**config.n_layers)
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
            self.config.res[0],
            self.config.res[1],
        )
        return z
