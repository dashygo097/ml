import torch
from torch import nn


class ScoreBasedRecommendHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        id: int,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        topk: int = 1,
    ) -> torch.Tensor:
        scores = emb1[id] @ emb2.T
        return scores.topk(topk, dim=-1).indices
