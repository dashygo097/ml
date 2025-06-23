from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from termcolor import colored
from torch import nn


class GCN(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        act: Callable = F.relu,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, colored(
            "[ERROR] Number of layers must be at least 2", "red", attrs=["bold"]
        )
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act

        self.conv_in = gnn.GCNConv(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [
                gnn.GCNConv(in_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers - 1)
            ]
        )
        self.conv_out = gnn.GCNConv(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv_in(x, edge_index, edge_weight)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_out(x, edge_index, edge_weight)
        return x
