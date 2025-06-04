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
        assert embed_size % n_heads == 0, (
            "[ERROR] embed_size must be divisible by n_heads"
        )
        self.d_model = embed_size
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads

        self.W_qkv = nn.Linear(embed_size, self.d_model * 3, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, C, E = x.shape
        Q, K, V = self.qkv(x)

        outputs, weights = scaled_dot_product_attention(Q, K, V, masked=mask)
        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        return self.dropout(outputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, E = x.shape

        QKV = self.W_qkv(x)
        Q, K, V = QKV.chunk(3, dim=-1)

        Q = Q.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)

        return Q, K, V
