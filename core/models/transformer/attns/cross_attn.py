from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .functional import scaled_dot_product_attention


@dataclass
class CrossAttnInfraRecord:
    input_logits: Tuple[torch.Tensor, ...]
    output_logits: Optional[torch.Tensor] = None
    attn_weights: Optional[torch.Tensor] = None
    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None

    def empty_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.attn_weights = None
        self.output_logits = None


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

    def prompt(self, record: CrossAttnInfraRecord) -> CrossAttnInfraRecord:
        x_1, x_2 = record.input_logits
        B, C_1, E = x_1.shape
        B, C_2, E = x_2.shape
        Q, K, V = self.qkv(x_1, x_2)
        outputs, weights = scaled_dot_product_attention(
            Q, K, V, mask="^", dropout=self.attn_dropout
        )
        outputs = (outputs.transpose(1, 2)).reshape(B, C_1, -1)
        outputs = self.W_o(outputs)
        record.k_cache = K
        record.v_cache = V
        record.attn_weights = weights
        record.output_logits = outputs
        return record

    def infer(
        self, record: CrossAttnInfraRecord, use_cache: bool = False
    ) -> CrossAttnInfraRecord:
        x_1, x_2 = record.input_logits
        B, C_1, E = x_1.shape
        B, C_2, E = x_2.shape
        if (
            use_cache
            and record.k_cache is not None
            and record.v_cache is not None
            and record.attn_weights is not None
        ):
            d_length = C_1 - record.k_cache.shape[2]
            new_inputs = x_1[:, -d_length:, :]
            Q, K, V = self.qkv(new_inputs, x_2)
            K = torch.cat([record.k_cache, K], dim=2)
            V = torch.cat([record.v_cache, V], dim=2)
            scores = Q @ K.transpose(-2, -1) / (self.head_dim**0.5)
            scores = torch.softmax(scores, dim=-1)
            weights = torch.cat(
                [
                    record.attn_weights,
                    torch.zeros(B, self.n_heads, C_1 - d_length, d_length),
                ],
                dim=-1,
            )
            weights = torch.cat([weights, scores], dim=2)
            outputs = weights @ V
            outputs = outputs.transpose(1, 2).reshape(B, C_1, -1)
            outputs = self.W_o(outputs)
            record.k_cache = K
            record.v_cache = V
            record.attn_weights = weights
            record.output_logits = outputs
            return record
        else:
            return self.prompt(record)

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
