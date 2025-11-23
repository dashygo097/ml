from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from ...trainer import TrainArgs, Trainer


class VAETrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]):
        super().__init__(path_or_dict)


class VAETrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        loss_fn: Callable,
        args: VAETrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(model, dataset, loss_fn, args, optimizer, scheduler, valid_ds)

    def step(
        self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]
    ) -> Dict[str, Any]:
        self.optimizer.zero_grad()
        batched = batch[0].to(self.device)

        output, mean, var = self.model(batched)
        output, mean, var = (
            output.to(self.device),
            mean.to(self.device),
            var.to(self.device),
        )
        loss = self.loss_fn(batched, output, mean, var)

        loss.backward()

        self.optimizer.step()

        return {"loss": loss.item()}
