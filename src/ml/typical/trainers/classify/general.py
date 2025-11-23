from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from termcolor import colored

from ....trainer import TrainArgs, Trainer
from ...data import BaseDataset


class ClassificationTrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)


class ClassificationTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: BaseDataset,
        loss_fn: Callable,
        args: TrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(model, dataset, loss_fn, args, optimizer, scheduler, valid_ds)

    def step(
        self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]
    ) -> Dict[str, Any]:
        self.optimizer.zero_grad()
        (inputs, targets), info = batch

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def validate(self) -> None:
        self.model.eval()
        total = 0
        total_loss = 0.0
        correct = 0

        if self.valid_data_loader is None:
            return

        for batch in self.valid_data_loader:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to(self.device)

                total += labels.size(0)
                total_loss += loss.item() * labels.size(0)

                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        self.logger.log(
            "valid", {"loss": avg_loss, "accuracy": accuracy}, index=self.n_epochs
        )
        print(
            f"(Validation {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {avg_loss}"
            + f", {colored('accuracy', 'green')}: {accuracy:.4f}"
        )


class ClassificationFinetuner(ClassificationTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: BaseDataset,
        loss_fn: Callable,
        args: TrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(model, dataset, loss_fn, args, optimizer, scheduler, valid_ds)
