from typing import Callable, List

import torch
from torch import nn


class ClassifyHead(nn.Module):
    def __init__(
        self,
        features: int | List[int],
        num_classes: int,
        act: Callable = nn.ReLU(),
        out_act: Callable = nn.Identity(),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        if isinstance(features, int):
            self.fc = nn.Linear(features, num_classes)
        elif isinstance(features, list):
            layers = []
            for i in range(len(features) - 1):
                layers.append(nn.Linear(features[i], features[i + 1]))
                if i < len(features) - 2:
                    layers.append(act)
                    layers.append(nn.Dropout(dropout))

            self.fc = nn.Sequential(*layers, nn.Linear(features[-1], num_classes))
        self.out_act = out_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.fc(x))


class ScoreBasedRecommendHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        user_id: int,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        topk: int = 1,
    ) -> List[int]:
        scores = user_emb[user_id] @ item_emb.T
        return scores.topk(topk, dim=-1).indices.tolist()
