from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..base import TrainArgs, Trainer


class ClassificationTrainer(Trainer):
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

        (inputs, targets), _ = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]
    ) -> Dict[str, Any]:
        (inputs, targets), _ = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.to(self.device)
            correct = (predicted == targets).sum().item()
            total = targets.size(0)

        return {"loss": loss.item(), "correct": correct, "total": total}
