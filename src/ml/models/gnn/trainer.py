from typing import Any, Callable, Dict, Optional

from termcolor import colored
from torch import nn

from ...data import BaseIterator
from ...trainer import TrainArgs, Trainer


class GNNTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        loss_fn: Callable,
        args: TrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(model, dataset, loss_fn, args, optimizer, scheduler, valid_ds)

    def set_dataset(self, dataset) -> None:
        self.data_loader = BaseIterator(dataset)

    def set_valid_ds(self, valid_ds) -> None:
        self.valid_data_loader = BaseIterator(valid_ds)

    def step(self, batch) -> Dict[str, Any]: ...

    def validate(self) -> None: ...

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
            + f": {self.logger.content.epoch[f'{self.n_epochs:}']['loss']:.4f}"
        )
