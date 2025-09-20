from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn

from ...base import GNNEncoder


class GraphMAE(GNNEncoder):
    def __init__(
        self,
        features: List[int],
        dropout: float = 0.5,
        act: Callable = F.relu,
        normalize: bool = True,
        residue: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.features = features
        self.in_features = features[0]
        self.out_features = features[-1]
        self.dropout = dropout
        self.act = act
        self.normalize = normalize
        self.num_layers = len(features) - 1
        self.residue = residue if residue is not None else [False] * self.num_layers

        self.enc = nn.ModuleList()
        self.res_project = nn.ModuleList()
        for i in range(self.num_layers):
            self.enc.append(
                gnn.GCNConv(features[i], features[i + 1], normalize=normalize)
            )

            if self.residue[i]:
                self.res_project.append(
                    nn.Linear(features[i], features[i + 1], bias=False)
                )
            else:
                self.res_project.append(None)

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
        for index, conv in enumerate(self.enc):
            if self.residue[index]:
                x = conv(x, edge_index, edge_attr) + self.res_project[index](x)
            else:
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

        for index, conv in enumerate(self.enc):
            if self.residue[index]:
                x = conv(x, edge_index) + self.res_project[index](x)
            else:
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
        for index, conv in enumerate(self.enc):
            if self.residue[index]:
                x = conv(x, edge_index, edge_attr) + self.res_project[index](x)
            else:
                x = conv(x, edge_index, edge_attr)
            feats.append(x)
            if apply_act:
                x = self.act(x)
            if apply_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return feats
