from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class AttnModel(ABC, nn.Module):
    def __init__(
        self, embed_size: int, d_model: Optional[int] = None, dropout: float = 0.1
    ):
        super().__init__()
        self.embed_size = embed_size
        self.d_model: int = d_model if d_model is not None else embed_size
        self.dropout = dropout

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]: ...
