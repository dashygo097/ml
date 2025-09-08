from typing import OrderedDict

import torch
from torch import nn

from ..minibatch import MiniBatch1d
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

    @torch.compile
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L_dim = z.shape
        z = z.view(B, L_dim).unsqueeze(-1).unsqueeze(-1)
        z = self.seq(z)

        return z


class DCGANDiscriminator(nn.Module):
    def __init__(self, config: DCGANConfig) -> None:
        super().__init__()
        self.config = config
        module_list = []
        fc_list = []
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
                    ("bn2d_" + str(index), nn.BatchNorm2d(out_channels)),
                    ("silu_" + str(index + 1), nn.SiLU(inplace=True)),
                    (
                        "dropout_" + str(index + 1),
                        nn.Dropout2d(self.config.dis_dropout),
                    ),
                ]
            )
            in_channels = out_channels
            out_channels = out_channels * 2
            img_shape = (
                (img_shape[0] + 1) // 2,
                (img_shape[1] + 1) // 2,
            )

        if self.config.dis_use_minibatch:
            fc_list.extend(
                [
                    ("flatten", nn.Flatten()),
                    (
                        "minibatch1d_0",
                        MiniBatch1d(
                            in_channels * img_shape[0] * img_shape[1],
                            self.config.dis_minibatch_dim,
                            self.config.dis_minibatch_inner_dim,
                        ),
                    ),
                    (
                        "linear_0",
                        nn.Linear(
                            in_channels * img_shape[0] * img_shape[1]
                            + self.config.dis_minibatch_dim,
                            self.config.dis_minibatch_out_features,
                        ),
                    ),
                    ("linear_1", nn.Linear(self.config.dis_minibatch_out_features, 1)),
                    ("act_0", nn.Sigmoid()),
                ]
            )

        else:
            fc_list.extend(
                [
                    (
                        "conv2d_out",
                        nn.Conv2d(
                            in_channels,
                            1,
                            kernel_size=self.config.dis_kernel_size,
                            stride=2,
                            padding=0,
                        ),
                    ),
                    ("act", nn.Sigmoid()),
                ]
            )
        self.seq = nn.Sequential(OrderedDict(module_list))
        self.fc = nn.Sequential(OrderedDict(fc_list))
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

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.seq(x)).reshape(x.shape[0], 1)
