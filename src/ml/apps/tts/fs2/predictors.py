from typing import OrderedDict

import torch
from torch import nn

from .config import FastSpeechConfig


class LayerNorm(nn.Module):
    def __init__(self, shape, eps: float = 1e-12) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x.transpose(1, -1))
        return x.transpose(1, -1)


class VariancePredictor(nn.Module):
    def __init__(self, config: FastSpeechConfig) -> None:
        super().__init__()

        self.embed_size = config.embed_size
        self.filter_size = config.adapter_filter_size
        self.kernel = config.adapter_kernel_size
        self.dropout = config.adapter_dropout

        self.seq = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        nn.Conv1d(
                            self.embed_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("leack_relu_1", nn.LeakyReLU(0.1)),
                    ("layernorm_1", LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        nn.Conv1d(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("leack_relu_2", nn.LeakyReLU(0.1)),
                    ("layernorm_2", LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.dense = nn.Linear(self.filter_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)

        x = self.seq(x)
        x = self.dense(x.transpose(1, -1)).squeeze(-1).long()
        return x
