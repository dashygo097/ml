from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import nn
from torch_geometric.data import Data


class GNNEncoder(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features: Optional[int] = None
        self.out_features: Optional[int] = None

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor: ...

    def feats(self, data: Data, *args, **kwargs) -> List[torch.Tensor]:
        return [self.forward(data)]
