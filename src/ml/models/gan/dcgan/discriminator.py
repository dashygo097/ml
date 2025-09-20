from typing import OrderedDict

import torch
from torch import nn

from ..components import MiniBatch1d
from .config import DCGANDiscriminatorConfig


class DCGANDiscriminator(nn.Module):
    def __init__(self, config: DCGANDiscriminatorConfig) -> None:
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
                        self.config.hidden_dim,
                        kernel_size=self.config.kernel_size,
                        stride=2,
                        padding=(self.config.kernel_size - 1) // 2,
                    ),
                ),
                ("silu_init", nn.SiLU(inplace=True)),
                ("dropout_init", nn.Dropout2d(self.config.dropout)),
            ]
        )
        in_channels = self.config.hidden_dim
        out_channels = self.config.hidden_dim * 2
        img_shape = ((self.config.res[0] + 1) // 2, self.config.res[1] // 2)

        for index in range(config.n_layers):
            module_list.extend(
                [
                    (
                        "conv2d_" + str(index + 1),
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=self.config.kernel_size,
                            stride=2,
                            padding=(self.config.kernel_size - 1) // 2,
                        ),
                    ),
                    ("bn2d_" + str(index), nn.BatchNorm2d(out_channels)),
                    ("silu_" + str(index + 1), nn.SiLU(inplace=True)),
                    (
                        "dropout_" + str(index + 1),
                        nn.Dropout2d(self.config.dropout),
                    ),
                ]
            )
            in_channels = out_channels
            out_channels = out_channels * 2
            img_shape = (
                (img_shape[0] + 1) // 2,
                (img_shape[1] + 1) // 2,
            )

        if self.config.use_minibatch:
            fc_list.extend(
                [
                    ("flatten", nn.Flatten()),
                    (
                        "minibatch1d_0",
                        MiniBatch1d(
                            in_channels * img_shape[0] * img_shape[1],
                            self.config.minibatch_dim,
                            self.config.minibatch_inner_dim,
                        ),
                    ),
                    (
                        "linear_0",
                        nn.Linear(
                            in_channels * img_shape[0] * img_shape[1]
                            + self.config.minibatch_dim,
                            self.config.minibatch_out_features,
                        ),
                    ),
                    ("linear_1", nn.Linear(self.config.minibatch_out_features, 1)),
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
                            kernel_size=self.config.kernel_size,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.seq(x)).reshape(x.shape[0], 1)
