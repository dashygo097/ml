from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import CrossAttnModel


class MulHeadCrossAttn(CrossAttnModel):
    def __init__(
        self,
        d_q: int,
        d_kv: int,
        n_heads: int,
        d_model: Optional[int] = None,
        bias: bool = False,
        enable_rope: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(d_q, d_kv, n_heads, d_model, bias, enable_rope, dropout)

        self.W_q = nn.Linear(self.d_q, self.d_model, bias=bias)
        self.W_kv = nn.Linear(self.d_kv, 2 * self.d_model, bias=bias)
        self.W_o = nn.Linear(self.d_model, self.d_q, bias=bias)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, E = x1.shape
        Q, K, V = self.qkv(x1, x2)

        outputs = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, dropout_p=self.dropout
        )
        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        return self.out_dropout(outputs)

    def qkv(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C1, E_1 = x1.shape
        B, C2, E_2 = x2.shape
        Q = self.W_q(x1)
        KV = self.W_kv(x2)
        K, V = KV.chunk(2, dim=-1)

        Q = Q.view(B, C1, self.n_heads, self.head_dim)
        K = K.view(B, C2, self.n_heads, self.head_dim)
        V = V.view(B, C2, self.n_heads, self.head_dim)

        if self.enable_rope:
            Q, K = self.rope(Q, K)

        return Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
