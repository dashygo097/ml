from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import AttnModel


class GroupedQueryAttn(AttnModel):
    def __init__(
        self,
        embed_size: int,
        q_heads: int,
        kv_heads: int,
        d_model: Optional[int] = None,
        bias: bool = False,
        enable_rope: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(embed_size, q_heads, d_model, bias, enable_rope, dropout)
        assert enable_rope is False, "RoPE not supported in GQA"

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.kv_embed_dim = self.d_model // q_heads * kv_heads
        self.heads_per_group = q_heads // kv_heads

        self.W_qkv = nn.Linear(
            self.embed_size, self.d_model + self.kv_embed_dim * 2, bias=bias
        )
        self.W_o = nn.Linear(self.d_model, self.embed_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, C, E = x.shape
        Q, K, V = self.qkv(x, pos)

        outputs = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
            enable_gqa=True,
        )
        outputs = outputs.transpose(1, 2).reshape(B, C, -1)
        outputs = self.W_o(outputs)
        return self.out_dropout(outputs)

    def qkv(
        self, x: torch.Tensor, pos: Optional[int] = None
    ) -> Tuple[torch.Tensor, ...]:
        B, C, E = x.shape
        QKV = self.W_qkv(x)
        Q = QKV[:, :, : self.d_model]
        K = QKV[:, :, self.d_model : self.d_model + self.kv_embed_dim]
        V = QKV[:, :, self.d_model + self.kv_embed_dim :]

        Q = Q.view(B, C, self.q_heads, self.head_dim)
        K = K.view(B, C, self.kv_heads, self.head_dim)
        V = V.view(B, C, self.kv_heads, self.head_dim)

        return Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
