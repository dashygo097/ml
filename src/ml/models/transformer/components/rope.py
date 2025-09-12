from typing import Optional, Tuple

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
        self.max_length = max_length

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_length, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer(
            "cache", torch.cat([freqs.cos(), freqs.sin()], dim=-1).unsqueeze_(1)
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, pos: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, D = x1.shape
        if pos is not None:
            if (C + pos) > self.max_length:
                x1 = x1[:, : self.max_length]
            cos, sin = self.cache[pos : min(self.max_length, pos + C)].chunk(2, dim=-1)
        else:
            if C > self.max_length:
                x1 = x1[:, : self.max_length]
            cos, sin = self.cache[:min(self.max_length, C)].chunk(2, dim=-1)

        x11, x12 = x1.chunk(2, dim=-1)
        x21, x22 = x2.chunk(2, dim=-1)

        x1 = torch.cat([x11 * cos - x12 * sin, x12 * cos + x11 * sin], dim=-1)
        x2 = torch.cat([x21 * cos - x22 * sin, x22 * cos + x21 * sin], dim=-1)

        return x1, x2
