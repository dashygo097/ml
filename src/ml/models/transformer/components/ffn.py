import torch
from torch import nn

from .swiglu import SwiGLU


class FFN(nn.Module):
    def __init__(self, d_model: int, d_inner: int, act=nn.GELU(), dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner

        self.linear1 = nn.Linear(d_model, d_inner)
        self.linear2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = act

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        return self.linear2(self.dropout(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_inner: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.linear1 = nn.Linear(d_model, d_inner)
        self.linear2 = nn.Linear(d_inner // 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = SwiGLU()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        return self.linear2(self.dropout(x))
