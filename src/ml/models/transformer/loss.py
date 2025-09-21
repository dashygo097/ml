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
        w_reg: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.w_cls = w_cls
        self.w_reg = w_reg
        self.alpha: float = alpha
        self.gamma: float = gamma

        self.cls_loss_fn: Callable = FocalLoss(alpha, gamma)

    def forward(
        self,
        pred_cls: torch.Tensor,
        pred_reg: torch.Tensor,
        tgt_cls: torch.Tensor,
        tgt_reg: torch.Tensor,
    ) -> torch.Tensor:
        pred_cls_flat = pred_cls.view(-1, pred_cls.shape[-1])
        tgt_cls_flat = tgt_cls.view(-1)

        cls_loss = self.cls_loss_fn(pred_cls_flat, tgt_cls_flat)

        pos_mask = (tgt_cls > 0).unsqueeze(-1).expand_as(pred_reg)
        pos_pred_reg = pred_reg[pos_mask].view(-1, pred_reg.shape[-1])
        pos_tgt_reg = tgt_reg[pos_mask].view(-1, tgt_reg.shape[-1])

        if pos_pred_reg.numel() > 0:
            pos_loss = F.smooth_l1_loss(pos_pred_reg, pos_tgt_reg, reduction="mean")
        else:
            pos_loss = torch.tensor(0.0, device=pred_reg.device)

        return self.w_cls * cls_loss + self.w_reg * pos_loss
