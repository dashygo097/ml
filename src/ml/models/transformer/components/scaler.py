import torch
from torch import nn


class LayerScaler(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.lambda_ = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda_
