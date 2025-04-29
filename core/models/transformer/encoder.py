from typing import OrderedDict, Tuple

import torch
import torch.nn as nn

from .attns import MulHeadAttn


class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(y))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_inner: int, act=nn.GELU(), dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner

        self.linear1 = nn.Linear(d_model, d_inner)
        self.linear2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        return self.linear2(self.dropout(x))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_inner: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner

        self.attn = MulHeadAttn(d_model, n_heads, dropout=dropout)
        self.ffn = FFN(d_model, d_inner, dropout=dropout)
        self.addnorm1 = AddNorm(d_model, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.addnorm1(x, self.attn(x, mask=mask))
        return self.addnorm2(x, self.ffn(x))

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.attn.qkv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_inner: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner

        module_list = []
        for index in range(n_layers):
            module_list.append(
                (
                    f"blk_{index}",
                    EncoderBlock(d_model, n_heads, d_inner, dropout),
                )
            )
        self.blks = nn.Sequential(OrderedDict(module_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blks:
            x = blk(x)
        return x

    def qkv(self, x: torch.Tensor):
        for index in range(self.n_layers - 1):
            x = self.blks[index](x)

        return list(self.blks._modules.values())[-1].qkv(x)  # pyright: ignore
