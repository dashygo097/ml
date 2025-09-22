from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import nn


class GNNEncoder(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = None
        self.out_features = None

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor: ...

    def feats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        *args,
        **kwargs,
    ) -> List[torch.Tensor]:
        return [self.forward(x, edge_index, edge_attr)]
