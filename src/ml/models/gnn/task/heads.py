from typing import Callable, List

import torch
from torch import nn

from ...mlp import MLP


class ClassifyHead(nn.Module):
    def __init__(
        self,
        features: int | List[int],
        num_classes: int,
        act: Callable = nn.ReLU(),
        out_act: Callable = nn.Identity(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = MLP(
            features=features,
            out_features=num_classes,
            act=act,
            out_act=out_act,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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
