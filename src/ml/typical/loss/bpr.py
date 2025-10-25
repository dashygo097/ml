import torch
from torch import nn


class BPRLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction: str = reduction

    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
    ) -> torch.Tensor:
        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
