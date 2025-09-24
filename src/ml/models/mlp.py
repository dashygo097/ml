from typing import Callable, List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        features: List[int],
        act: Callable = nn.Identity(),
        out_act: Callable = nn.Identity(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(features) >= 2, "Features list must contain at least one element."
        self.in_features = features[0]
        self.out_features = features[-1]
        self.dropout = dropout

        layers = []
        for i in range(len(features) - 2):
            layers.append(nn.Linear(features[i], features[i + 1]))
            if i < len(features) - 2:
                layers.append(act)
                layers.append(nn.Dropout(dropout))

        self.fc = nn.Sequential(*layers, nn.Linear(features[-2], self.out_features))
        self.out_act = out_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.fc(x))
