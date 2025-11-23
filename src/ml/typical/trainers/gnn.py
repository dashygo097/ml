from typing import Any, Callable, Dict, Optional

from torch import nn

from ...iterator import BaseIterator
from ...trainer import TrainArgs, Trainer


class GNNTrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict) -> None:
        super().__init__(path_or_dict)


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
