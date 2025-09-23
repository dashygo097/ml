from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from termcolor import colored

from ...trainer import TrainArgs, Trainer


class CNNTrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)


class CNNTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion: Callable,
        args: TrainArgs,
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
        inputs, targets = batch

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
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
                loss = self.criterion(outputs, labels)

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


class CNNFinetuner(CNNTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion: Callable,
        args: TrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )

    def step_info(self, result: Dict[str, Any]) -> None:
        self.logger.op(
            "step",
            lambda x: {"loss": x.get("loss", 0) + result["loss"]},
            index=self.n_steps,
        )
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) + result["loss"]},
            index=self.n_epochs,
        )
