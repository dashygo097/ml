from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from termcolor import colored
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
        criterion: Callable,
        args: VAETrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )

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
        loss = self.criterion(batched, output, mean, var)

        loss.backward()

        self.optimizer.step()

        return {"loss": loss.item()}

    def step_info(self, result: Dict[str, Any]) -> None:
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) + result["loss"]},
            index=self.n_epochs,
        )

    def epoch_info(self) -> None:
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) / len(self.data_loader)},
            index=self.n_epochs,
        )
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['loss']}"
        )
