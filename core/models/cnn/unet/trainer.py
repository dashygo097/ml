from typing import Dict

import torch.nn as nn

from ....trainer import Trainer, TrainerArgs


class UnetTrainArgs(TrainerArgs):
    def __init__(self, path: str) -> None:
        super().__init__(path)


class UnetTrainer(Trainer):
    def __init__(
        self, model: nn.Module, dataset, criterion, args: TrainerArgs, optimizer=None
    ) -> None:
        super().__init__(model, dataset, criterion, args, optimizer)

    def step(self, batch):
        self.optimizer.zero_grad()
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss}

    def step_info(self, result: Dict) -> None:
        if f"epoch {self.n_epochs}" not in self.logger:
            self.logger[f"epoch {self.n_epochs}"] = {}
            self.logger[f"epoch {self.n_epochs}"]["loss"] = 0.0

        self.logger[f"epoch {self.n_epochs}"]["loss"] += float(result["loss"])

    def epoch_info(self) -> None:
        self.logger[f"epoch {self.n_epochs}"]["loss"] /= (
            len(self.data_loader) * self.args.batch_size
        )
