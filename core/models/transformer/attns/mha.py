from typing import Tuple

import torch
from torch import nn

from .functional import scaled_dot_product_attention


class MulHeadAttn(nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.d_model = embed_size
        self.n_heads = n_heads

        self.W_q = nn.Linear(embed_size, self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)

        self.W_k = nn.Linear(embed_size, self.d_model, bias=False)
        self.W_v = nn.Linear(embed_size, self.d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, C, E = x.shape

        Q = self.W_q(x)

        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(B, C, self.d_model // self.n_heads, self.n_heads).transpose(1, 2)
        K = K.view(B, C, self.d_model // self.n_heads, self.n_heads).transpose(1, 2)
        V = V.view(B, C, self.d_model // self.n_heads, self.n_heads).transpose(1, 2)

        outputs, weights = scaled_dot_product_attention(Q, K, V, masked=mask)
        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        return self.dropout(outputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, E = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(B, C, self.d_model // self.n_heads, self.n_heads).transpose(1, 2)
        K = K.view(B, C, self.d_model // self.n_heads, self.n_heads).transpose(1, 2)
        V = V.view(B, C, self.d_model // self.n_heads, self.n_heads).transpose(1, 2)
        return Q, K, V


class CrossAttn(nn.Module):
    def __init__(
        self,
        embed_size: int,
        d_q: int,
        d_kv: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.d_q = d_q
        self.d_kv = d_kv

        self.W_q = nn.Linear(embed_size, self.d_q)
        self.W_k = nn.Linear(embed_size, self.d_kv)
        self.W_v = nn.Linear(embed_size, self.d_kv)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor, mask=None) -> torch.Tensor:
        B, C, E = x_2.shape
        Q = self.W_q(x_1)
        K = self.W_k(x_2)
        V = self.W_v(x_2)

        outputs, weights = scaled_dot_product_attention(Q, K, V, masked=mask)
        return self.dropout(outputs)

    def qkv(
        self, x_1: torch.Tensor, x_2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, E = x_2.shape
        Q = self.W_q(x_1)
        K = self.W_k(x_2)
        V = self.W_v(x_2)
        return Q, K, V


class MulHeadCrossAttn(nn.Module):
    def __init__(
        self,
        embed_size: int,
        context_size: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.d_q = context_size
        self.d_kv = context_size
        self.n_heads = n_heads

        self.W_q = nn.Linear(embed_size, self.d_q, bias=False)
        self.W_o = nn.Linear(self.d_kv, self.d_kv, bias=False)

        self.W_k = nn.Linear(embed_size, self.d_kv, bias=False)
        self.W_v = nn.Linear(embed_size, self.d_kv, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor, mask=None) -> torch.Tensor:
        _, C, E = x_1.shape
        B, C_2, E_2 = x_2.shape

        Q = self.W_q(x_1)

        K = self.W_k(x_2)
        V = self.W_v(x_2)
        Q = Q.view(B, C, self.d_q // self.n_heads, self.n_heads).transpose(1, 2)
        K = K.view(B, C_2, self.d_kv // self.n_heads, self.n_heads).transpose(1, 2)
        V = V.view(B, C_2, self.d_kv // self.n_heads, self.n_heads).transpose(1, 2)

        outputs, weights = scaled_dot_product_attention(Q, K, V, masked=mask)
        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        return self.dropout(outputs)

    def qkv(
        self, x_1: torch.Tensor, x_2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, C, E = x_1.shape
        B, C_2, E_2 = x_2.shape
        Q = self.W_q(x_1)
        K = self.W_k(x_2)
        V = self.W_v(x_2)
        Q = Q.view(B, C, self.d_q // self.n_heads, self.n_heads).transpose(1, 2)
        K = K.view(B, C_2, self.d_kv // self.n_heads, self.n_heads).transpose(1, 2)
        V = V.view(B, C_2, self.d_kv // self.n_heads, self.n_heads).transpose(1, 2)
        return Q, K, V
