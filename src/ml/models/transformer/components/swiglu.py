import torch
from torch import nn


def swiglu(x: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * torch.nn.functional.silu(x2)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x)
