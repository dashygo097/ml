from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn

from ..base import GNNEncoder


class GATBackbone(GNNEncoder):
    def __init__(
        self,
        features: List[int],
        heads: int,
        act: Callable = F.relu,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_layers = len(features)
        self.features = features
        self.in_features = features[0]
        self.out_features = features[-1]
        self.act = act
        self.dropout = dropout

        convs = []
        for i in range(self.num_layers - 1):
            convs.extend(
                [
                    gnn.GATv2Conv(
                        features[i], features[i + 1], heads=heads, concat=False
                    ),
                ]
            )

        self.convs = nn.ModuleList(convs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edge_weight = edge_attr.float() if edge_attr is not None else None

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def feats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        apply_act: bool = False,
        apply_dropout: bool = True,
    ) -> List[torch.Tensor]:
        feats = []
        edge_weight = edge_attr.float() if edge_attr is not None else None

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            feats.append(x)
            if apply_act:
                x = self.act(x)
            if apply_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return feats
