from typing import Optional, Tuple

import torch
from torch import nn

from .base import AttnModel
from .functional import scaled_dot_product_attention


class MulHeadAttn2d(AttnModel):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        d_model: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_size, d_model, dropout)
        self.n_heads = n_heads
        self.head_dim = self.d_model // n_heads

        self.W_qkv = nn.Conv2d(
            in_channels=self.embed_size,
            out_channels=self.d_model * 3,
            kernel_size=1,
            bias=False,
        )

        self.W_o = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.embed_size,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[str] | torch.Tensor = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        Q, K, V = self.qkv(x)

        outputs = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.attn_dropout
        )
        outputs = outputs.view(B, self.d_model, H, W)
        outputs = self.W_o(outputs)

        return self.out_dropout(outputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, H, W = x.shape

        QKV = self.W_qkv(x)
        Q, K, V = torch.chunk(QKV, chunks=3, dim=1)

        Q = Q.view(B, self.n_heads, self.head_dim, H * W)
        K = K.view(B, self.n_heads, self.head_dim, H * W)
        V = V.view(B, self.n_heads, self.head_dim, H * W)

        return Q, K, V
