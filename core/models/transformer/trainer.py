from typing import Dict

import torch.nn as nn

from ...trainer import Trainer, TrainerArgs


class GPTrainerArgs(TrainerArgs):
    def __init__(self, path: str) -> None:
        super().__init__(path)


class GPTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion,
        args: GPTrainerArgs,
        optimizer=None,
    ) -> None:
        super().__init__(model, dataset, criterion, args, optimizer)

    def step(self, batch) -> Dict:
        return {}
