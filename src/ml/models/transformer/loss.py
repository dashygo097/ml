from typing import Callable

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
        w_cls: float = 1.0,
        w_bbox: float = 1.0,
        w_angle: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.w_cls = w_cls
        self.w_bbox = w_bbox
        self.w_angle = w_angle
        self.alpha: float = alpha
        self.gamma: float = gamma

        self.cls_loss_fn: Callable = FocalLoss(alpha, gamma)

    def forward(
        self,
        pred_cls: torch.Tensor,
        pred_bbox: torch.Tensor,
        pred_angle: torch.Tensor,
        tgt_cls: torch.Tensor,
        tgt_bbox: torch.Tensor,
        tgt_angle: torch.Tensor,
    ) -> torch.Tensor:
        pred_cls_flat = pred_cls.view(-1, pred_cls.shape[-1])
        tgt_cls_flat = tgt_cls.view(-1)
        cls_loss = self.cls_loss_fn(pred_cls_flat, tgt_cls_flat)

        pos_mask = tgt_cls > 0

        if pos_mask.sum() > 0:
            pos_pred_bbox = pred_bbox[pos_mask]
            pos_pred_angle = pred_angle[pos_mask]
            pos_tgt_bbox = tgt_bbox[pos_mask]
            pos_tgt_angle = tgt_angle[pos_mask]

            bbox_loss = F.smooth_l1_loss(pos_pred_bbox, pos_tgt_bbox, reduction="mean")
            angle_loss = F.smooth_l1_loss(
                pos_pred_angle, pos_tgt_angle, reduction="mean"
            )
        else:
            bbox_loss = torch.tensor(0.0, device=pred_bbox.device)
            angle_loss = torch.tensor(0.0, device=pred_angle.device)

        return (
            self.w_cls * cls_loss + self.w_bbox * bbox_loss + self.w_angle * angle_loss
        )
