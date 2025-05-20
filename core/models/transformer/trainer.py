from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ...trainer import TrainArgs, Trainer


class GPTrainArgs(TrainArgs):
    def __init__(self, path: str) -> None:
        super().__init__(path)


class GPTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion,
        args: GPTrainArgs,
        optimizer=None,
    ) -> None:
        super().__init__(model, dataset, criterion, args, optimizer)

    def step(self, batch) -> Dict:
        return {}
