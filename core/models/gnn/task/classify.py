from typing import Callable

import torch
from termcolor import colored
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from ..base import GNNEncoder
from .heads import GNNClassifyHead


class GNNClassifier(nn.Module):
    def __init__(
        self,
        encoder: GNNEncoder,
        n_classes: int,
        level: str = "node",
        fusion: Callable = global_mean_pool,
    ) -> None:
        super().__init__()
        if encoder.out_features is None:
            raise ValueError(
                colored(
                    "[ERROR] Encoder must have a defined output feature size.",
                    color="red",
                    attrs=["bold"],
                )
            )

        if level == "node":
            print(
                colored(
                    "[WARN] Using node-level classification, fusion layer ignored",
                    color="yellow",
                    attrs=["bold"],
                )
            )

        self.n_classes = n_classes
        self.level = level

        self.encoder = encoder
        self.head = GNNClassifyHead(encoder.out_features, n_classes)
        self.fusion = fusion

    def forward(self, data: Data) -> torch.Tensor:
        if self.level == "node":
            x = self.encoder(data)
            return self.head(x)

        elif self.level == "graph":
            x = self.encoder(data)
            x = self.fusion(x, data.batch)
            return self.head(x)

        else:
            raise ValueError(
                colored(
                    f"[ERROR] Unsupported level: {self.level}. Use 'node' or 'graph'.)",
                    color="red",
                    attrs=["bold"],
                )
            )
