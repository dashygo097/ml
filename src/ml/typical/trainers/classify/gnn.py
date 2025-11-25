from typing import Any, Callable, Dict, Optional

import torch
from termcolor import colored
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dropout_edge

from ....models import GNNEncoder
from ...data import BaseIterator
from ...heads import ClassifyHead
from ..base import TrainArgs, Trainer


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


class GNNClassifyTrainer(Trainer):
    def __init__(
        self,
        model: GNNClassifier,
        train_ds: Any,
        loss_fn: Callable,
        args: TrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(model, train_ds, loss_fn, args, optimizer, scheduler, valid_ds)

    def _setup_dataloaders(self, train_ds: Any, valid_ds: Optional[Any]) -> None:
        self.train_loader = BaseIterator(train_ds)
        self.val_loader = BaseIterator(valid_ds) if valid_ds else None

    def step(self, batch: Any) -> Dict[str, float]:
        self.optimizer.zero_grad()

        out = self.model(batch.to(self.device))
        if hasattr(batch, "train_mask"):
            loss = self.loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
        else:
            loss = self.loss_fn(out, batch.y)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch: Any) -> Dict[str, float]:
        batch = batch.to(self.device)
        out = self.model(batch)
        loss = self.loss_fn(out[batch.val_mask], batch.y[batch.val_mask])

        preds = out.argmax(dim=-1)
        correct = (preds[batch.val_mask] == batch.y[batch.val_mask]).sum().item()

        return {"val_loss": loss.item(), "correct": correct}
