from typing import Tuple

import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.cos_cache = None
        self.sin_cache = None

    def _build(self, x: torch.Tensor) -> None:
        C = x.shape[1]
        position = torch.arange(C, device=x.device).float()
        sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        self.cos_cache = sinusoid_inp.cos()
        self.sin_cache = sinusoid_inp.sin()

    def _rotate_2(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return x1, x2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.cos_cache is None
            or self.sin_cache is None
            or self.cos_cache.shape[0] != x.shape[1]
        ):
            self._build(x)

        x1, x2 = self._rotate_2(x)
        x = torch.cat(
            [
                x1 * self.cos_cache - x2 * self.sin_cache,  # pyright: ignore
                x1 * self.sin_cache + x2 * self.cos_cache,  # pyright: ignore
            ],
            dim=-1,
        )

        return x
