from typing import Dict

from termcolor import colored
from torch_geometric.utils import dropout_edge

from ..base_trainer import GNNTrainer, TrainArgs
from .classify import GNNClassifier


class GNNClassifierTrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict) -> None:
        super().__init__(path_or_dict)
        self.patience = self.args.get("patience", 10)
        self.min_delta = self.args.get("min_delta", 0.001)
        self.edge_dropout = self.args.get("edge_dropout", 0.0)


class GNNClassifyTrainer(GNNTrainer):
    def __init__(
        self,
        model: GNNClassifier,
        dataset,
        criterion,
        args: GNNClassifierTrainArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )

        self._best_val_loss = float("inf")
        self._no_improve_epochs = 0

    def step(self, batch) -> Dict:
        self.optimizer.zero_grad()

        batch.edge_index, _ = dropout_edge(
            batch.edge_index, p=self.args.edge_dropout, training=self.model.training
        )

        out = self.model(batch.to(self.device))
        if hasattr(batch, "train_mask"):
            loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
        else:
            loss = self.criterion(out, batch.y)

        """
        # KD loss
        if self.teacher is not None:
            with torch.no_grad():
                t_out = self.teacher(batch.to(self.device))
            kd_loss = F.kl_div(
                F.log_softmax(out, dim=-1),
                F.softmax(t_out, dim=-1),
                reduction="batchmean",
            )
            loss = loss * (1 - alpha) + kd_loss * alpha

        """

        loss.backward()
        self.optimizer.step()

        return {"loss": loss}

    def validate(self) -> None:
        self.model.eval()
        total_loss, total_correct, total_val = 0, 0, 0

        for data in self.valid_data_loader:
            data = data.to(self.device)
            out = self.model(data)
            loss = self.criterion(out[data.val_mask], data.y[data.val_mask])
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
            self._stop_training = True
