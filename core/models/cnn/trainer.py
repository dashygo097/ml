from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from termcolor import colored

from ...trainer import TrainArgs, Trainer


class CNNTrainArgs(TrainArgs):
    def __init__(self, path: str) -> None:
        super().__init__(path)


class CNNTrainer(Trainer):
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

    def step(self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]) -> Dict:
        self.optimizer.zero_grad()
        inputs, targets = batch

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss}

    def step_info(self, result: Dict) -> None:
        epoch_logger = self.logger["epoch"]
        if f"epoch {self.n_epochs}" not in epoch_logger:
            epoch_logger[f"epoch {self.n_epochs}"] = {}
            epoch_logger[f"epoch {self.n_epochs}"]["loss"] = 0.0

        epoch_logger[f"epoch {self.n_epochs}"]["loss"] += float(result["loss"].sum())
        self.logger["epoch"] = epoch_logger

    def epoch_info(self) -> None:
        self.logger["epoch"][f"epoch {self.n_epochs}"]["loss"] /= len(self.data_loader)
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {self.logger['epoch'][f'epoch {self.n_epochs}']['loss']}"
        )

        if self.n_epochs % 20 == 0 and self.n_epochs > 0:
            self.save()

        self.save_log(info=False)

    def validate(self) -> None:
        self.model.eval()
        total = 0
        total_loss = 0.0
        correct = 0

        if self.valid_data_loader is None:
            print("No validation dataset provided.")
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
        self.logger["valid"][f"epoch {self.n_epochs}"] = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
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
        criterion,
        args: TrainArgs,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__(model, dataset, criterion, args, optimizer, scheduler)

    def step_info(self, result: Dict) -> None:
        step_logger = self.logger["step"]
        epoch_logger = self.logger["epoch"]
        if f"epoch {self.n_epochs}" not in epoch_logger:
            epoch_logger[f"epoch {self.n_epochs}"] = {}
            epoch_logger[f"epoch {self.n_epochs}"]["loss"] = 0.0
        epoch_logger[f"epoch {self.n_epochs}"]["loss"] += float(result["loss"].sum())

        if f"step {self.n_steps}" not in step_logger:
            step_logger[f"step {self.n_steps}"] = {}
            step_logger[f"step {self.n_steps}"]["loss"] = 0.0

        step_logger[f"step {self.n_steps}"]["loss"] = float(result["loss"].sum())
