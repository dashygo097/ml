from typing import Callable, Optional

import torch
from termcolor import colored
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from ..heads import ClassifyHead
from .base import GNNEncoder


class GNNClassifier(nn.Module):
    def __init__(
        self,
        encoder: GNNEncoder,
        num_classes: int,
        level: str = "node",
        fusion: Callable = global_mean_pool,
        head: Optional[nn.Module] = None,
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

        self.num_classes = num_classes
        self.level = level

        self.encoder = encoder
        self.head = (
            head
            if head is not None
            else ClassifyHead(encoder.out_features, num_classes=num_classes)
        )
        self.fusion = fusion

    def forward(self, data: Data) -> torch.Tensor:
        return self.head(self.encode(data))

    def predict(self, data: Data) -> torch.Tensor:
        logits = self.forward(data)
        return logits.argmax(dim=-1)

    def encode(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.get("edge_attr", None)
        if self.level == "node":
            return self.encoder(x, edge_index, edge_attr)
        elif self.level == "graph":
            x = self.encoder(x, edge_index, edge_attr)
            return self.fusion(x, batch=data.batch)
        else:
            raise ValueError(
                colored(
                    f"[ERROR] Unsupported level: {self.level}. Use 'node' or 'graph'.)",
                    color="red",
                    attrs=["bold"],
                )
            )
