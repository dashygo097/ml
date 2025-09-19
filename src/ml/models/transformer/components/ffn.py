from typing import Callable

import torch
from torch import nn

from .swiglu import SwiGLU


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        act: Callable = nn.GELU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.act = act

        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.fc2(self.dropout(x))


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.act = SwiGLU()

        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner // 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.fc2(self.dropout(x))
