from typing import Callable, List

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from termcolor import colored
from torch_geometric.data import Data

from ..base import GNNEncoder


class GCNBackBone(GNNEncoder):
    def __init__(
        self,
        features: List[int],
        act: Callable = F.relu,
        dropout: float = 0.5,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = len(features)
        self.features = features
        self.in_features = features[0]
        self.out_features = features[-1]
        self.act = act
        self.dropout = dropout
        self.normalize = normalize

        self.convs = []
        for i in range(self.num_layers - 1):
            self.convs.extend(
                [
                    (
                        f"blk_{i}",
                        gnn.GCNConv(features[i], features[i + 1], normalize=normalize),
                    ),
                ]
            )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if x is None:
            raise ValueError(
                colored(
                    "[ERROR] data.x should NOT be None", color="red", attrs=["bold"]
                )
            )
        return x

    def feats(
        self,
        data: Data,
        apply_act: bool = False,
        apply_dropout: bool = True,
    ) -> List[torch.Tensor]:
        feats = []

        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            feats.append(x)
            if apply_act:
                x = self.act(x)
            if apply_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return feats
