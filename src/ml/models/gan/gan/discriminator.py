from typing import OrderedDict

from torch import nn

from ..minibatch import MiniBatch1d
from .config import GANConfig


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
                ("dropout_0", nn.Dropout(self.config.dis_dropout)),
            ]
        )

        for index in range(config.n_dis_layers):
            in_dim = self.config.dis_hidden_dim // (2**index)
            module_list.extend(
                [
                    ("linear_" + str(index + 1), nn.Linear(in_dim, in_dim // 2)),
                    ("leaky_relu_" + str(index + 1), nn.LeakyReLU(0.2, inplace=True)),
                    (
                        "dropout_" + str(index + 1),
                        nn.Dropout(self.config.dis_dropout),
                    ),
                ]
            )

        fit_dim = self.config.dis_hidden_dim // (2**config.n_dis_layers)
        if self.config.dis_use_minibatch:
            module_list.extend(
                [
                    (
                        "minibatch",
                        MiniBatch1d(
                            fit_dim,
                            self.config.dis_minibatch_dim,
                            self.config.dis_minibatch_inner_dim,
                        ),
                    ),
                    ("dense", nn.Linear(fit_dim + self.config.dis_minibatch_dim, 1)),
                    ("act", nn.Sigmoid()),
                ]
            )
        else:
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
        return x
