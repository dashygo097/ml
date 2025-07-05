from typing import Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class BipartiteGraphDataset(Dataset):
    def __init__(self, data: Dict, directional: bool = True) -> None:
        self.graph = Data(
            edge_index=data["edge_index"],
            edge_attr=data["edge_attr"],
        )
        self.label = Data(
            edge_index=data["edge_label_index"],
            edge_attr=data["edge_label"],
        )

        if not directional:
            self.make_undirected(self.graph)
            self.make_undirected(self.label)

    def make_undirected(self, data: Data) -> None:
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        data.edge_index = edge_index
        data.edge_attr = edge_attr

    def __len__(self) -> int:
        return self.label.edge_index.shape[1]

    def __getitem__(self, idx: int) -> Data:
        edge_index = self.label.edge_index[:, idx]
        edge_attr = self.label.edge_attr[idx]
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
