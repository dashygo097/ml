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
        act: Callable = F.elu,
        dropout: float = 0.3,
        normalize: bool = True,
        residue: Optional[List[bool]] = None,
    ) -> None:
        super().__init__()
        self.num_layers = len(features)
        self.features = features
        self.in_features = features[0]
        self.out_features = features[-1]
        self.act = act
        self.dropout = dropout
        self.residue = (
            residue if residue is not None else [False] * (self.num_layers - 1)
        )

        self.convs = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.extend(
                [
                    gnn.GATv2Conv(
                        features[i], features[i + 1] // heads, heads=heads, concat=True
                    ),
                ]
            )

            if normalize:
                self.norms.append(gnn.LayerNorm(features[i + 1]))

            if self.residue[i]:
                self.res_proj.append(
                    nn.Linear(features[i], features[i + 1], bias=False)
                )
            else:
                self.res_proj.append(None)

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edge_weight = edge_attr.float() if edge_attr is not None else None

        for index, conv in enumerate(self.convs):
            if self.residue[index]:
                x = conv(x, edge_index, edge_weight) + self.res_proj[index](x)
            else:
                x = conv(x, edge_index, edge_weight)

            if self.norms:
                x = self.norms[index](x)

            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    @torch.compile
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

        for index, conv in enumerate(self.convs):
            if self.residue[index]:
                x = conv(x, edge_index, edge_weight) + self.res_proj[index](x)
            else:
                x = conv(x, edge_index, edge_weight)
            feats.append(x)
            if apply_act:
                x = self.act(x)
            if apply_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return feats
