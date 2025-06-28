from typing import List

import torch
from torch import nn


class GNNClassifyHead(nn.Module):
    def __init__(self, out_features: int, n_classes: int) -> None:
        super().__init__()
        self.n_classes
        self.fc = nn.Linear(out_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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
