from typing import Optional, Tuple

import torch
from torch import nn

from .functional import scaled_dot_product_attention


class CrossAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_q: int,
        d_kv: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_q = d_q
        self.d_kv = d_kv

        self.W_q = nn.Linear(self.d_q, self.d_model, bias=False)
        self.W_kv = nn.Linear(self.d_kv, 2 * self.d_model, bias=False)

        self.W_o = nn.Linear(self.d_model, self.d_q, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        mask: Optional[str] | torch.Tensor = None,
    ) -> torch.Tensor:
        B, C, E = x_1.shape
        Q, K, V = self.qkv(x_1, x_2)

        outputs = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.attn_dropout
        )
        outputs = outputs.transpose(1, 2).reshape(B, C, -1)
        outputs = self.W_o(outputs)
        return self.out_dropout(outputs)

    def qkv(self, x_1: torch.Tensor, x_2: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        Q = self.W_q(x_1)
        KV = self.W_kv(x_2)
        K, V = KV.chunk(2, dim=-1)

        Q = Q.unsqueeze(1)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        return Q, K, V


class MulHeadCrossAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_q: int,
        d_kv: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.d_q = d_q
        self.d_kv = d_kv
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(self.d_q, self.d_model, bias=False)
        self.W_kv = nn.Linear(self.d_kv, 2 * self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_q, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor, mask=None) -> torch.Tensor:
        B, C, E = x_1.shape
        Q, K, V = self.qkv(x_1, x_2)

        outputs = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.attn_dropout
        )
        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        return self.out_dropout(outputs)

    def qkv(
        self, x_1: torch.Tensor, x_2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C_1, E = x_1.shape
        B, C_2, E = x_2.shape
        Q = self.W_q(x_1)
        KV = self.W_kv(x_2)
        K, V = KV.chunk(2, dim=-1)

        Q = Q.view(B, C_1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, C_2, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, C_2, self.n_heads, self.head_dim).transpose(1, 2)

        return Q, K, V
