from typing import List, Tuple

import torch
from torch import nn

from ....models import MLP


class MLPObjDetectionHead2D(nn.Module):
    def __init__(
        self,
        features: int | List[int],
        num_classes: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.features = features
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.dropout = dropout

        bbox_features = features + [4] if isinstance(features, List) else [features, 4]
        cls_features = (
            features + [num_classes + 1]
            if isinstance(features, List)
            else [features, num_classes + 1]
        )
        self.bbox_head = MLP(bbox_features, dropout=dropout)
        self.cls_head = MLP(cls_features, dropout=dropout)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bbox_preds = self.bbox_head(features).sigmoid()
        cls_logits = self.cls_head(features)
        return cls_logits, bbox_preds
