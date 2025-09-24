from typing import Any, Callable, Dict

import torch
from termcolor import colored
from torch import nn

from ...data import ChangeDetectionDataset
from ...trainer import TrainArgs, Trainer


class ChangeDetectionTrainerArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)


class ChangeDetectionTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: ChangeDetectionDataset,
        criterion: Callable,
        args: ChangeDetectionTrainerArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )

    def step(self, batch) -> Dict[str, Any]:
        (imgs1, imgs2, labels), info = batch
        imgs1, imgs2, labels = (
            imgs1.to(self.device),
            imgs2.to(self.device),
            labels.to(self.device),
        )
        self.optimizer.zero_grad()
        logits = self.model(imgs1, imgs2)
        loss = self.criterion(logits, labels)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def step_info(self, result: Dict[str, Any]) -> None:
        if self.n_steps % 10 == 0 and self.n_steps > 0:
            self.logger.op(
                "step",
                lambda x: {"loss": x.get("loss", 0) + result["loss"]},
                index=self.n_steps,
            )
            print(
                f"(Step {self.n_steps}) "
                + colored("loss", "yellow")
                + f": {self.logger.content.step[f'{self.n_steps}']['loss']:.4f}"
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
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['loss']:.4f}"
        )

    def validate(self) -> None:
        self.model.eval()
        total_loss, total_correct, total_num_labels, total_val = 0, 0, 0, 0

        for batch in self.valid_data_loader:
            (imgs1, imgs2, labels), info = batch
            imgs1, imgs2, labels = (
                imgs1.to(self.device),
                imgs2.to(self.device),
                labels.to(self.device),
            )
            with torch.no_grad():
                logits = self.model(imgs1, imgs2)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_num_labels += labels.numel()
                total_val += labels.size(0)

        val_loss = total_loss / total_val
        val_acc = total_correct / total_num_labels

        self.logger.log(
            "valid", {"val_loss": val_loss, "val_acc": val_acc}, self.n_epochs
        )
        print(
            f"(Validation {self.n_epochs}) "
            + f" {colored('loss', 'red')}: {val_loss:.4f} "
            + f", {colored('accuracy', 'green')}: {val_acc:.4f}"
        )
