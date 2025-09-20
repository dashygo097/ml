from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData


class BipartiteGNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        in_features: Tuple[int, int],
        hiddens: Tuple[int, int],
        node_types: Tuple[str, str] = ("node_type_0", "node_type_1"),
        relation_type: str = "relates",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hiddens = hiddens
        self.node_types = node_types
        self.relation_type = relation_type

        self.backbone = backbone
        self.emb = nn.ModuleList()
        for i in range(len(in_features)):
            self.emb.append(nn.Linear(in_features[i], hiddens[i]))

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = []
        for i in range(len(self.in_features)):
            x.append(self.emb[i](data[self.node_types[i]].x))
        edge_index = data[
            self.node_types[0], self.relation_type, self.node_types[-1]
        ].edge_index
        edge_attr = data[
            self.node_types[0], self.relation_type, self.node_types[-1]
        ].edge_attr
        out = self.backbone(tuple(x), edge_index, edge_attr)
        return out
