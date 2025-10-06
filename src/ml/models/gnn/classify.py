from typing import Any, Callable, Dict, Optional

import torch
from termcolor import colored
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dropout_edge

from ..heads import ClassifyHead
from .model import GNNEncoder
from .trainer import GNNTrainer, TrainArgs


class GNNClassifier(nn.Module):
    def __init__(
        self,
        encoder: GNNEncoder,
        num_classes: int,
        level: str = "node",
        fusion: Callable = global_mean_pool,
        head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if encoder.out_features is None:
            raise ValueError(
                colored(
                    "[ERROR] Encoder must have a defined output feature size.",
                    color="red",
                    attrs=["bold"],
                )
            )

        if level == "node":
            print(
                colored(
                    "[WARN] Using node-level classification, fusion layer ignored",
                    color="yellow",
                    attrs=["bold"],
                )
            )

        self.num_classes = num_classes
        self.level = level

        self.encoder = encoder
        self.head = (
            head
            if head is not None
            else ClassifyHead(encoder.out_features, num_classes=num_classes)
        )
        self.fusion = fusion

    def forward(self, data: Data) -> torch.Tensor:
        return self.head(self.encode(data))

    def predict(self, data: Data) -> torch.Tensor:
        logits = self.forward(data)
        return logits.argmax(dim=-1)

    def encode(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.get("edge_attr", None)
        if self.level == "node":
            return self.encoder(x, edge_index, edge_attr)
        elif self.level == "graph":
            x = self.encoder(x, edge_index, edge_attr)
            return self.fusion(x, batch=data.batch)
        else:
            raise ValueError(
                colored(
                    f"[ERROR] Unsupported level: {self.level}. Use 'node' or 'graph'.)",
                    color="red",
                    attrs=["bold"],
                )
            )


class GNNClassifierTrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)
        self.patience: int = self.args.get("patience", 10)
        self.min_delta: float = self.args.get("min_delta", 0.001)
        self.edge_dropout: float = self.args.get("edge_dropout", 0.0)


class GNNClassifyTrainer(GNNTrainer):
    def __init__(
        self,
        model: GNNClassifier,
        dataset,
        loss_fn: Callable,
        args: GNNClassifierTrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(model, dataset, loss_fn, args, optimizer, scheduler, valid_ds)

        self._best_val_loss = float("inf")
        self._no_improve_epochs = 0

    def step(self, batch) -> Dict[str, Any]:
        self.optimizer.zero_grad()

        batch.edge_index, _ = dropout_edge(
            batch.edge_index, p=self.args.edge_dropout, training=self.model.training
        )

        out = self.model(batch.to(self.device))
        if hasattr(batch, "train_mask"):
            loss = self.loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
        else:
            loss = self.loss_fn(out, batch.y)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate(self) -> None:
        self.model.eval()
        total_loss, total_correct, total_val = 0, 0, 0

        for data in self.valid_data_loader:
            data = data.to(self.device)
            out = self.model(data)
            loss = self.loss_fn(out[data.val_mask], data.y[data.val_mask])
            total_loss += loss.item() * data.val_mask.sum().item()

            preds = out.argmax(dim=-1)
            total_correct += (
                (preds[data.val_mask] == data.y[data.val_mask]).sum().item()
            )
            total_val += data.val_mask.sum().item()

        val_loss = total_loss / total_val
        val_acc = total_correct / total_val

        if val_loss < self._best_val_loss - self.args.min_delta:
            self._best_val_loss = val_loss
            self._no_improve_epochs = 0
        else:
            self._no_improve_epochs += 1

        self.logger.log(
            "valid", {"val_loss": val_loss, "val_acc": val_acc}, self.n_epochs
        )
        print(
            f"(Validation {self.n_epochs}) "
            + f" {colored('loss', 'red')}: {val_loss:.4f} "
            + f", {colored('accuracy', 'green')}: {val_acc:.4f}"
        )

        if self._no_improve_epochs >= self.args.patience:
            print(
                f"Early stopping triggered after {self._no_improve_epochs} unimproved epochs."
            )
            self.should_stop()
