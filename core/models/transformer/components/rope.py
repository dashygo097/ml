import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None

    def _build(self, seq_len: int, device: torch.device) -> None:
        pos = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        self.cos_cache = freqs.cos()[None, None, :, :]
        self.sin_cache = freqs.sin()[None, None, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, S, D = x.shape
        assert D == self.dim, f"Expected last dim={self.dim}, got {D}"

        if (
            self.cos_cache is None
            or self.sin_cache is None
            or self.cos_cache.shape[2] < S
        ):
            self._build(S, x.device)

        cos = self.cos_cache[:, :, :S, :]
        sin = self.sin_cache[:, :, :S, :]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return x
