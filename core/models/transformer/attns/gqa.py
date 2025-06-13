import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import AttnInfraRecord, AttnModel
from .functional import scaled_dot_product_attention, sdp_attn


class GroupedQueryAttn(AttnModel):
    def __init__(
        self,
        embed_size: int,
        q_heads: int,
        kv_heads: int,
        d_model: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_size, d_model, dropout)
        assert self.d_model % q_heads == 0, (
            "[ERROR] embed_size must be divisible by n_heads"
        )

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.kv_embed_dim = self.d_model // q_heads * kv_heads
        self.head_dim = self.d_model // q_heads
        self.heads_per_group = q_heads // kv_heads

        self.W_q = nn.Linear(self.embed_size, self.d_model, bias=False)
        self.W_kv = nn.Linear(self.embed_size, self.kv_embed_dim * 2, bias=False)
        self.W_o = nn.Linear(self.d_model, self.embed_size)

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
        outputs = outputs.permute(0, 3, 1, 2, 4).reshape(
            B, C, self.heads_per_group * self.kv_heads * self.head_dim
        )
        outputs = self.W_o(outputs)
        return self.out_dropout(outputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, E = x.shape
        Q = (
            self.W_q(x)
            .view(B, C, self.heads_per_group, self.kv_heads, self.head_dim)
            .permute(0, 2, 3, 1, 4)
        )
        KV = self.W_kv(x).view(B, C, self.kv_heads, 2, self.head_dim).transpose(1, 2)
        K, V = KV.unbind(dim=3)
        K = K.unsqueeze(1).expand(-1, self.heads_per_group, -1, -1, -1)
        V = V.unsqueeze(1).expand(-1, self.heads_per_group, -1, -1, -1)

        return Q, K, V

    def prompt(self, record: AttnInfraRecord) -> AttnInfraRecord:
        B, C, E = record.input_logits.shape
        Q, K, V = self.qkv(record.input_logits)
        outputs, weights = sdp_attn(Q, K, V, mask="^")
        outputs = outputs.permute(0, 3, 1, 2, 4).reshape(
            B, C, self.heads_per_group * self.kv_heads * self.head_dim
        )
        outputs = self.W_o(outputs)
        record.k_cache = K
        record.v_cache = V
        record.attn_weights = weights
        record.output_logits = outputs
        return record

    def infer(
        self, record: AttnInfraRecord, use_cache: bool = False
    ) -> AttnInfraRecord:
        B, C, E = record.input_logits.shape

        if (
            use_cache
            and record.k_cache is not None
            and record.v_cache is not None
            and record.attn_weights is not None
        ):
            d_length = C - record.k_cache.shape[2]
            new_inputs = record.input_logits[:, -d_length:, :]

            Q, K, V = self.qkv(new_inputs)

            K = torch.cat([record.k_cache, K], dim=2)
            V = torch.cat([record.v_cache, V], dim=2)

            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = F.softmax(scores, dim=-1)
            weights = torch.cat(
                [
                    record.attn_weights,
                    torch.zeros(B, self.q_heads, C - d_length, d_length),
                ],
                dim=-1,
            )
            weights = torch.cat([weights, scores], dim=2)

            outputs = weights @ V
            outputs = outputs.transpose(1, 2).reshape(B, C, -1)
            outputs = self.W_o(outputs)

            record.k_cache = K
            record.v_cache = V
            record.attn_weights = weights
            record.output_logits = outputs
            return record

        else:
            return self.prompt(record)
