from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

from .base import TrainArgs, Trainer


class VAETrainer(Trainer):
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

    def step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]
    ) -> Dict[str, Any]:
        self.optimizer.zero_grad()

        (imgs, _), _ = batch
        imgs = imgs.to(self.device)

        output, mean, var = self.model(imgs)
        output, mean, var = (
            output.to(self.device),
            mean.to(self.device),
            var.to(self.device),
        )
        loss = self.loss_fn(imgs, output, mean, var)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
