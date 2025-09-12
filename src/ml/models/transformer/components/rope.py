from typing import Tuple

import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000,
        max_length: int = 1024,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_length, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer(
            "cache", torch.cat([freqs.cos(), freqs.sin()], dim=-1).unsqueeze_(1)
        )

    def forward_same_qk(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, D = x1.shape
        cos, sin = self.cache[:C].chunk(2, dim=-1)

        x11, x12 = x1.chunk(2, dim=-1)
        x21, x22 = x2.chunk(2, dim=-1)

        x1 = torch.cat([x11 * cos - x12 * sin, x12 * cos + x11 * sin], dim=-1)
        x2 = torch.cat([x21 * cos - x22 * sin, x22 * cos + x21 * sin], dim=-1)

        return x1, x2

    def forward_diff_qk(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Lq, H, D = x1.shape
        _, Lk, _, _ = x2.shape

        cos_q, sin_q = self.cache[:Lq].chunk(2, dim=-1)
        cos_k, sin_k = self.cache[:Lk].chunk(2, dim=-1)

        x11, x12 = x1.chunk(2, dim=-1)
        x21, x22 = x2.chunk(2, dim=-1)

        x1 = torch.cat([x11 * cos_q - x12 * sin_q, x12 * cos_q + x11 * sin_q], dim=-1)
        x2 = torch.cat([x21 * cos_k - x22 * sin_k, x22 * cos_k + x21 * sin_k], dim=-1)

        return x1, x2

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x1.shape[1] == x2.shape[1]:
            return self.forward_same_qk(x1, x2)
        else:
            return self.forward_diff_qk(x1, x2)
