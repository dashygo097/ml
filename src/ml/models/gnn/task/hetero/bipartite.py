from typing import List

import torch
from torch import nn
from torch_geometric.data import HeteroData


class BipartiteGNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        in_features: List[int],
        hidden: int,
    ) -> None:
        super().__init__()
        self.features = in_features
        self.hidden = hidden

        self.backbone = backbone
        self.emb = nn.ModuleList()
        for i in range(len(in_features)):
            self.emb.append(nn.Linear(in_features[i], hidden))

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = []
        for i in range(len(self.features)):
            x.append(self.emb[i](data[f"node_type_{i}"].x))
        edge_index = data["node_type_0", "relates", "node_type_1"].edge_index
        edge_attr = data["node_type_0", "relates", "node_type_1"].edge_attr
        out = self.backbone(tuple(x), edge_index, edge_attr)
        return out
