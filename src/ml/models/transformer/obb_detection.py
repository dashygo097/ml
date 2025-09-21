from typing import Any, Callable, Dict, Tuple

import torch
from termcolor import colored
from torch import nn

from ...trainer import TrainArgs, Trainer
from .loss import OBBLoss


class OBBDetectionTrainerArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)


class OBBDetectionTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion: Callable,
        args: OBBDetectionTrainerArgs,
        collect_fn: Callable = lambda x: x,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )
        self.collect_fn: Callable = collect_fn

    def set_dataset(self, dataset) -> None:
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.is_shuffle,
            num_workers=self.args.num_workers,
            collate_fn=self.collect_fn,
        )

    def step(self, batch: Tuple[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]: ...

    def step_info(self, result: Dict[str, Any]) -> None:
        # step
        if self.n_steps % 1000 == 0:
            self.logger.op(
                "step",
                lambda x: {"loss": x.get("loss", 0) + result["loss"]},
                index=self.n_steps,
            )
            print(
                f"(Step {self.n_steps}) "
                + colored("loss", "yellow")
                + f": {self.logger.content.step[f'{self.n_steps}']['loss']}"
            )

        # epoch
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

    def validate(self) -> None: ...
