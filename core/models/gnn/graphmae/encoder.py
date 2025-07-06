from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn

from ..base import GNNEncoder


class GraphMAE(GNNEncoder):
    def __init__(
        self,
        features: List[int],
        dropout: float = 0.5,
        act: Callable = F.relu,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.features = features
        self.in_features = features[0]
        self.out_features = features[-1]
        self.dropout = dropout
        self.act = act
        self.normalize = normalize

        self.num_layers = len(features) - 1

        self.enc = nn.ModuleList()
        for i in range(self.num_layers):
            self.enc.append(
                gnn.GCNConv(features[i], features[i + 1], normalize=normalize)
            )

        self.dec = nn.Sequential(
            nn.Linear(features[-1], features[-2]),
            nn.ReLU(),
            nn.Linear(features[-2], features[0]),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for conv in self.enc:
            x = conv(x, edge_index, edge_attr)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reconstruct(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_original = x.clone()
        x[mask] = 0.0

        for conv in self.enc:
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_rec = self.dec(x[mask])
        x_target = x_original[mask]
        return x_rec, x_target

    def feats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        apply_act: bool = False,
        apply_dropout: bool = True,
    ) -> List[torch.Tensor]:
        feats = []
        for conv in self.enc:
            x = conv(x, edge_index, edge_attr)
            feats.append(x)
            if apply_act:
                x = self.act(x)
            if apply_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return feats
