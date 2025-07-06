from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attns import AttnModel, MulHeadAttn
from .components import FFN, AddNorm


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_inner: int,
        dropout: float = 0.1,
        attn: Optional[AttnModel] = None,
        ffn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner

        self.attn = (
            MulHeadAttn(d_model, n_heads, dropout=dropout) if attn is None else attn
        )
        self.ffn = FFN(d_model, d_inner, dropout=dropout) if ffn is None else ffn
        self.addnorm1 = AddNorm(d_model, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout=dropout)

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
