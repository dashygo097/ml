from typing import Dict

from torch import nn

from ..base_trainer import GNNTrainer, TrainArgs


class GNNClassifyTrainer(GNNTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion,
        args: TrainArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )

    def step(self, batch) -> Dict: ...

    def validate(self) -> None: ...
