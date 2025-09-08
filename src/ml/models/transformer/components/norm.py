import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        element_affine: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.element_affine = element_affine
        if element_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(
            torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps
        )
        if self.element_affine:
            x = self.weight * x
        return x


class AddNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    @torch.compile
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(y))
