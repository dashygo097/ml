from typing import Optional

import torch
import torch.nn as nn

from .attns import AttnModel, MulHeadAttn, MulHeadCrossAttn
from .ffn import FFN, AddNorm


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_inner: int,
        dropout: float = 0.1,
        attn: Optional[nn.Module] = None,
        ffn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner

        self.attn = (
            MulHeadAttn(d_model, n_heads, dropout=dropout) if attn is None else attn
        )
        self.attn_mask = MulHeadCrossAttn(
            d_model, n_heads, d_q=d_model, d_kv=d_model, dropout=dropout
        )
        self.ffn = FFN(d_model, d_inner, dropout=dropout) if ffn is None else ffn
        self.addnorm1 = AddNorm(d_model, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout=dropout)
        self.addnorm3 = AddNorm(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        enc_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ):
        x = self.addnorm1(x, self.attn(x, mask=mask, is_causal=is_causal))
        x = self.addnorm2(x, self.attn_mask(x, enc, mask=enc_mask))
        return self.addnorm3(x, self.ffn(x))


class DecoderOnlyBlock(nn.Module):
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
        is_causal: bool = True,
    ) -> torch.Tensor:
        x = self.addnorm1(x, self.attn(x, mask=mask, is_causal=is_causal))
        return self.addnorm2(x, self.ffn(x))
