import torch
from torch import nn


class ClassifyHead(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
