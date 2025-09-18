from typing import Optional

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(
            torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps
        )
        if self.element_affine:
            x = self.weight * x
        return x


class AddPostNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        norm: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if norm is not None:
            self.norm = norm
        else:
            eps = kwargs.pop("eps", 1e-6)
            element_affine = kwargs.pop("element_affine", True)
            self.norm = RMSNorm(d_model, eps=eps, element_affine=element_affine)
        dropout = kwargs.pop("dropout", 0.0)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_attn: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.out_dropout(x_attn))


class AddPreNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        norm: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if norm is not None:
            self.norm = norm
        else:
            eps = kwargs.pop("eps", 1e-6)
            element_affine = kwargs.pop("element_affine", True)
            self.norm = RMSNorm(d_model, eps=eps, element_affine=element_affine)
        dropout = kwargs.pop("dropout", 0.0)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_attn: torch.Tensor) -> torch.Tensor:
        return x + self.out_dropout(self.norm(x_attn))
