from typing import Callable, Dict

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class OBBLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma

        self.cls_loss_fn: Callable = FocalLoss()

    def forward(
        self, preds: Dict[str, torch.Tensor], tgts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred_cls = preds["pred_cls"]
        pred_pos = preds["pred_reg"][..., :4]
        pred_angle = preds["pred_reg"][..., 4]
        tgt_cls = tgts["tgt_cls"]
        tgt_pos = tgts["tgt_reg"][..., :4]
        tgt_angle = tgts["tgt_reg"][..., 4]

        pred_angle_sincos = torch.stack(
            (torch.sin(pred_angle), torch.cos(pred_angle)), dim=-1
        )
        tgts_angle_sincos = torch.stack(
            (torch.sin(tgt_angle), torch.cos(tgt_angle)), dim=-1
        )

        cls_loss = self.cls_loss_fn(pred_cls, tgt_cls)
        pos_loss = F.smooth_l1_loss(pred_pos, tgt_pos, reduction="mean")
        angle_loss = F.smooth_l1_loss(
            pred_angle_sincos, tgts_angle_sincos, reduction="mean"
        )

        return self.alpha * cls_loss + self.beta * pos_loss + self.gamma * angle_loss
