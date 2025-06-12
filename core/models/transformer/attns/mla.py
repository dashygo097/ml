from typing import Optional, Tuple

import torch
from torch import nn

from ..rope import RoPE
from .functional import scaled_dot_product_attention


class MulHeadLatentAttn(nn.Module):
    # NOTE: Confirm latend_dim << head_dim * num_heads
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        latent_dim: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_size % num_heads == 0, (
            "[ERROR] embed_size must be divisible by n_heads"
        )
        self.d_model = embed_size
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = head_dim if head_dim is not None else embed_size // num_heads
        self.split_dim = self.head_dim // 2

        self.W_dkv = nn.Linear(embed_size, latent_dim, bias=False)
        self.W_kv = nn.Linear(
            latent_dim, self.head_dim * num_heads // 2 * 3, bias=False
        )
        self.W_kr = nn.Linear(embed_size, self.head_dim * num_heads // 2, bias=False)

        self.W_dq = nn.Linear(embed_size, self.head_dim * num_heads, bias=False)
        self.W_q = nn.Linear(
            self.head_dim * num_heads, self.head_dim * num_heads, bias=False
        )

        self.W_o = nn.Linear(self.head_dim * num_heads, self.d_model, bias=False)
        self.rope = RoPE(self.head_dim // 2)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[str] | torch.Tensor = None,
    ) -> torch.Tensor:
        B, C, E = x.shape
        Q, K, V = self.qkv(x)

        outputs = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.attn_dropout
        )

        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        return self.out_dropout(outputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, E = x.shape
        KR = self.rope(
            self.W_kr(x).view(B, C, self.num_heads, self.split_dim).transpose(1, 2)
        )
        K, V, V_ = self.W_kv(self.W_dkv(x)).chunk(3, dim=-1)
        K = torch.cat(
            [K.view(B, C, self.num_heads, self.split_dim).transpose(1, 2), KR], dim=-1
        )
        V = (
            torch.cat([V, V_], dim=-1)
            .view(B, C, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        Q, Q_ = torch.chunk(self.W_q(self.W_dq(x)), 2, dim=-1)
        Q = Q.view(B, C, self.num_heads, self.split_dim).transpose(1, 2)
        Q_ = Q_.view(B, C, self.num_heads, self.split_dim).transpose(1, 2)
        Q = torch.cat([Q, self.rope(Q_)], dim=-1)

        return Q, K, V
