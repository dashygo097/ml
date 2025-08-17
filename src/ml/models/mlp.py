from typing import Callable, List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        features: int | List[int],
        out_features: int,
        act: Callable = nn.ReLU(),
        out_act: Callable = nn.Identity(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = out_features
        self.dropout = dropout

        if isinstance(features, int):
            self.fc = nn.Linear(features, out_features)
        elif isinstance(features, list):
            layers = []
            for i in range(len(features) - 1):
                layers.append(nn.Linear(features[i], features[i + 1]))
                if i < len(features) - 2:
                    layers.append(act)
                    layers.append(nn.Dropout(dropout))

            self.fc = nn.Sequential(*layers, nn.Linear(features[-1], out_features))
        self.out_act = out_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.fc(x))
