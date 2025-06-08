import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .functional import scaled_dot_product_attention, sdp_attn


@dataclass
class InfraRecord:
    input_logits: torch.Tensor
    output_logits: Optional[torch.Tensor] = None
    attn_weights: Optional[torch.Tensor] = None
    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None


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

    def forward(
        self,
        x: torch.Tensor,
        mask=None,
    ) -> torch.Tensor:
        B, C, E = x.shape
        Q, K, V = self.qkv(x)

        outputs = scaled_dot_product_attention(Q, K, V, mask=mask)

        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        return self.dropout(outputs)

    def prompt(self, inputs: Dict) -> Tuple[Dict, Dict]:
        assert "logits" in inputs.keys(), "[ERROR] inputs must contain 'logits' key"
        B, C, E = inputs["logits"].shape
        Q, K, V = self.qkv(inputs["logits"])

        outputs, weights, scores = sdp_attn(Q, K, V, mask="^")
        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        inputs["cache"] = {"k": K, "v": V, "weights": weights}
        return inputs, {"logits": outputs}

    def infer(self, inputs: Dict, use_cache: bool = False) -> Tuple[Dict, Dict]:
        assert "logits" in inputs.keys(), "[ERROR] inputs must contain 'logits' key"
        B, C, E = inputs["logits"].shape

        if use_cache and "cache" in inputs.keys():
            assert inputs["cache"]["k"].shape[2] == inputs["cache"]["v"].shape[2], (
                "[ERROR] cache must have the same length for k and v"
            )

            d_length = C - inputs["cache"]["k"].shape[2]
            new_inputs = inputs["logits"][:, -d_length:, :]

            Q, K, V = self.qkv(new_inputs)

            K = torch.cat([inputs["cache"]["k"], K], dim=2)
            V = torch.cat([inputs["cache"]["v"], V], dim=2)

            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = F.softmax(scores, dim=-1)
            weights = torch.cat(
                [
                    inputs["cache"]["weights"],
                    torch.zeros(B, self.n_heads, C - d_length, d_length),
                ],
                dim=-1,
            )
            weights = torch.cat([weights, scores], dim=2)

            outputs = weights @ V
            outputs = outputs.transpose(1, 2).reshape(B, C, -1)
            outputs = self.W_o(outputs)

            inputs["cache"] = {"k": K, "v": V, "weights": weights}
            return inputs, {"logits": outputs}

        else:
            return self.prompt(inputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, E = x.shape

        QKV = self.W_qkv(x)
        Q, K, V = QKV.chunk(3, dim=-1)

        Q = Q.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)

        return Q, K, V
