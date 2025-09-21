from typing import Any, Callable, Dict, Tuple

import torch
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
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )

    def set_dataset(self, dataset) -> None: ...

    def step(self, batch: Tuple[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]: ...
    def step_info(self, result: Dict[str, Any]) -> None: ...
    def epoch_info(self) -> None: ...
    def validate(self) -> None: ...
