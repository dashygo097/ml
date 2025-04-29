import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        epsilon: float = 1e-6,
        element_affine: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = epsilon
        self.element_affine = element_affine
        if element_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(
            torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps
        )
        if self.element_affine:
            x = self.weight * x
        return x
