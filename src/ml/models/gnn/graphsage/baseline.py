import inspect
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn

from ..base import GNNEncoder


class GraphSAGEBackbone(GNNEncoder):
    def __init__(
        self,
        features: List[int],
        aggr: type[gnn.Aggregation] = gnn.MeanAggregation,
        aggr_learn: bool = False,
        act: Callable = F.relu,
        dropout: float = 0.3,
        normalize: bool = True,
        residue: Optional[List[bool]] = None,
    ) -> None:
        super().__init__()
        self.num_layers = len(features)
        self.features = features
        self.aggr = aggr
        self.aggr_learn = aggr_learn
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
            aggr_kwargs = {}

            if "learn" in inspect.signature(aggr.__init__).parameters:
                aggr_kwargs["learn"] = aggr_learn

            self.convs.append(
                gnn.SAGEConv(features[i], features[i + 1], aggr=aggr(**aggr_kwargs))
            )

            if normalize:
                self.norms.append(gnn.LayerNorm(features[i + 1]))

            if self.residue[i]:
                self.res_proj.append(
                    nn.Linear(features[i], features[i + 1], bias=False)
                )
            else:
                self.res_proj.append(None)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,  # unused for SAGE
    ) -> torch.Tensor:
        for index, conv in enumerate(self.convs):
            out = conv(x, edge_index)
            if self.residue[index]:
                out = out + self.res_proj[index](x)
            x = out
            if self.norms:
                x = self.norms[index](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,  # unused
        apply_act: bool = False,
        apply_dropout: bool = True,
    ) -> List[torch.Tensor]:
        feats = []
        for index, conv in enumerate(self.convs):
            out = conv(x, edge_index)
            if self.residue[index]:
                out = out + self.res_proj[index](x)
            x = out
            feats.append(x)
            if apply_act:
                x = self.act(x)
            if apply_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return feats
