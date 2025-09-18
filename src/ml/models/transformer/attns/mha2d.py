from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import AttnModel


class MulHeadAttn2d(AttnModel):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        d_model: Optional[int] = None,
        bias: bool = False,
        enable_rope: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(embed_size, n_heads, d_model, bias, enable_rope, dropout)
        self.W_qkv = nn.Conv2d(
            in_channels=self.embed_size,
            out_channels=self.d_model * 3,
            kernel_size=1,
            bias=bias,
        )
        self.W_o = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.embed_size,
            kernel_size=1,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        Q, K, V = self.qkv(x, pos)

        outputs = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, dropout_p=self.dropout, is_causal=is_causal
        )
        outputs = outputs.view(B, self.d_model, H, W)
        outputs = self.W_o(outputs)

        return self.out_dropout(outputs)

    def qkv(
        self, x: torch.Tensor, pos: Optional[int] = None
    ) -> Tuple[torch.Tensor, ...]:
        B, C, H, W = x.shape

        QKV = self.W_qkv(x)
        Q, K, V = torch.chunk(QKV, chunks=3, dim=1)

        Q = Q.view(B, self.n_heads, self.head_dim, H * W).permute(0, 3, 1, 2)
        K = K.view(B, self.n_heads, self.head_dim, H * W).permute(0, 3, 1, 2)
        V = V.view(B, self.n_heads, self.head_dim, H * W).permute(0, 3, 1, 2)

        if self.enable_rope:
            Q, K = self.rope(Q, K, pos)

        return Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
