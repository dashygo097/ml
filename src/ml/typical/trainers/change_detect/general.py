from typing import Any, Callable, Dict, Tuple

import torch
from torch import nn

from ...data import ChangeDetectionDataset
from ..base import TrainArgs, Trainer


class ChangeDetectionTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: ChangeDetectionDataset,
        loss_fn: Callable,
        args: TrainArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(model, dataset, loss_fn, args, optimizer, scheduler, valid_ds)

    def step(self, batch: Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]) -> Dict[str, Any]:
        self.optimizer.zero_grad()

        (imgs1, imgs2, labels), _ = batch
        imgs1, imgs2, labels = (
            imgs1.to(self.device),
            imgs2.to(self.device),
            labels.to(self.device),
        )

        logits = self.model(imgs1, imgs2)
        loss = self.loss_fn(logits, labels)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]) -> Dict[str, Any]:
        (imgs1, imgs2, labels), _ = batch
        imgs1, imgs2, labels = (
            imgs1.to(self.device),
            imgs2.to(self.device),
            labels.to(self.device),
        )

        with torch.no_grad():
            logits = self.model(imgs1, imgs2)
            loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            num_labels = labels.numel()

        return {
            "val_loss": loss.item(),
            "correct": correct,
            "num_labels": num_labels,
            "batch_size": labels.size(0),
        }
