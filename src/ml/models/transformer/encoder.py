from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attns import AttnModel, MulHeadAttn
from .components import FFN, AddNorm


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        d_inner: Optional[int] = None,
        d_model: Optional[int] = None,
        attn: Optional[AttnModel] = None,
        ffn: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.d_inner = d_inner
        self.d_model = d_model if d_model is not None else embed_size
        self.d_inner = d_inner if d_inner is not None else 4 * self.d_model
        self.bias = bias
        self.dropout = dropout

        self.attn = (
            MulHeadAttn(
                embed_size, n_heads, d_model=d_model, bias=bias, dropout=dropout
            )
            if attn is None
            else attn
        )
        self.ffn = (
            FFN(self.d_model, self.d_inner, dropout=dropout) if ffn is None else ffn
        )
        self.addnorm1 = AddNorm(self.d_model, norm=norm, dropout=dropout)
        self.addnorm2 = AddNorm(self.d_model, norm=norm, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.addnorm1(x, self.attn(x, mask=mask, is_causal=is_causal))
        return self.addnorm2(x, self.ffn(x))

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.attn.qkv(x)
