from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

from ..base import TrainArgs, Trainer


class DepthEstTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_ds: Any,
        loss_fn: Callable,
        args: TrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(model, train_ds, loss_fn, args, optimizer, scheduler, valid_ds)

    def step(self, batch: Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]) -> Dict[str, Any]:
        self.optimizer.zero_grad()

        (imgs, labels), _ = batch
        imgs, labels = (
            imgs.to(self.device),
            labels.to(self.device),
        )

        logits = self.model(imgs)
        loss = self.loss_fn(logits, labels)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]) -> Dict[str, Any]:
        (imgs, labels), _ = batch
        imgs, labels = (
            imgs.to(self.device),
            labels.to(self.device),
        )

        with torch.no_grad():
            preds = self.model(imgs)
            loss = self.loss_fn(preds, labels)

        return {"val_loss": loss.item()}
