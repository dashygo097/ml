from typing import Optional, Tuple

import torch
from torch import nn

from .functional import scaled_dot_product_attention


class GroupedQueryAttn(nn.Module):
    def __init__(
        self, embed_size: int, q_heads: int, kv_heads: int, dropout: float = 0.1
    ) -> None:
        assert embed_size % q_heads == 0, (
            "[ERROR] embed_size must be divisible by n_heads"
        )

        super().__init__()
        self.embed_size = embed_size
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.kv_embed_dim = embed_size // q_heads * kv_heads
        self.head_dim = embed_size // q_heads
        self.heads_per_group = q_heads // kv_heads

        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_kv = nn.Linear(embed_size, self.kv_embed_dim * 2, bias=False)
        self.W_o = nn.Linear(embed_size, embed_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[str] | torch.Tensor = None,
    ) -> torch.Tensor:
        B, C, E = x.shape
        Q, K, V = self.qkv(x)

        outputs = scaled_dot_product_attention(Q, K, V, mask=mask)
        outputs = outputs.permute(0, 3, 1, 2, 4).reshape(
            B, C, self.heads_per_group * self.kv_heads * self.head_dim
        )
        outputs = self.W_o(outputs)
        return self.dropout(outputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, E = x.shape
        Q = (
            self.W_q(x)
            .view(B, C, self.heads_per_group, self.kv_heads, self.head_dim)
            .permute(0, 2, 3, 1, 4)
        )
        KV = self.W_kv(x).view(B, C, self.kv_heads, 2, self.head_dim).transpose(1, 2)
        K, V = KV.unbind(dim=3)

        return Q, K, V
