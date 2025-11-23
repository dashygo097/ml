from typing import Any, Callable, Dict

import torch
from termcolor import colored
from torch import nn

from ....trainer import TrainArgs, Trainer
from ...data import ChangeDetectionDataset


class ChangeDetectionTrainerArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)


class ChangeDetectionTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: ChangeDetectionDataset,
        loss_fn: Callable,
        args: ChangeDetectionTrainerArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(model, dataset, loss_fn, args, optimizer, scheduler, valid_ds)

    def step(self, batch) -> Dict[str, Any]:
        (imgs1, imgs2, labels), info = batch
        imgs1, imgs2, labels = (
            imgs1.to(self.device),
            imgs2.to(self.device),
            labels.to(self.device),
        )
        self.optimizer.zero_grad()
        logits = self.model(imgs1, imgs2)
        loss = self.loss_fn(logits, labels)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

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
                loss = self.loss_fn(logits, labels)
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
