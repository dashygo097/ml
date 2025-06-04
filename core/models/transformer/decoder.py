from typing import Optional, OrderedDict

import torch
import torch.nn as nn

from .attns import MulHeadAttn, MulHeadCrossAttn
from .encoder import FFN, AddNorm
from .rope import RoPEWithMaxLength


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_inner: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner

        self.attn = MulHeadAttn(d_model, n_heads, dropout=dropout)
        self.attn_mask = MulHeadCrossAttn(d_model, d_model, n_heads, dropout=dropout)
        self.ffn = FFN(d_model, d_inner, dropout=dropout)
        self.addnorm1 = AddNorm(d_model, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout=dropout)
        self.addnorm3 = AddNorm(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        mask: Optional[str] | torch.Tensor = "^",
        enc_mask: Optional[str] | torch.Tensor = None,
    ):
        x = self.addnorm1(x, self.attn(x, mask=mask))
        x = self.addnorm2(x, self.attn_mask(x, enc, mask=enc_mask))
        return self.addnorm3(x, self.ffn(x))


class Decoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_inner: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner
        self.n_layers = n_layers

        module_list = []
        for index in range(self.n_layers):
            module_list.append(
                (
                    f"blk_{index}",
                    DecoderBlock(d_model, n_heads, d_inner, dropout=dropout),
                )
            )
        self.blks = nn.Sequential(OrderedDict(module_list))

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        mask: Optional[str] | torch.Tensor = "^",
        enc_mask: Optional[str] | torch.Tensor = None,
    ) -> torch.Tensor:
        for blk in self.blks:
            x = blk(x, enc, mask=mask, enc_mask=enc_mask)
        return x


class DecoderOnlyBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_inner: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner

        self.attn = MulHeadAttn(d_model, n_heads, dropout=dropout)
        self.ffn = FFN(d_model, d_inner, dropout=dropout)
        self.addnorm1 = AddNorm(d_model, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[str] | torch.Tensor = "^",
    ) -> torch.Tensor:
        x = self.addnorm1(x, self.attn(x, mask=mask))
        return self.addnorm2(x, self.ffn(x))


class DecoderOnlyBody(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_inner: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner

        module_list = []
        for index in range(self.n_layers):
            module_list.append(
                (
                    f"blk_{index}",
                    DecoderOnlyBlock(d_model, n_heads, d_inner, dropout=dropout),
                )
            )

        self.blks = nn.Sequential(OrderedDict(module_list))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[str] | torch.Tensor = "^",
    ) -> torch.Tensor:
        for blk in self.blks:
            x = blk(x, mask=mask)
        return x


class DecoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_inner: int,
        dropout: float = 0.1,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = d_inner
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RoPEWithMaxLength(d_model, max_length=max_length)

        self.body = DecoderOnlyBody(
            n_layers, d_model, n_heads, d_inner, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dense = nn.Linear(d_model, vocab_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[str] | torch.Tensor = "^"
    ) -> torch.Tensor:
        x = self.embedding(x)
        x = self.rope(x)
        x = self.body(x, mask=mask)
        return self.dense(self.norm(x))
