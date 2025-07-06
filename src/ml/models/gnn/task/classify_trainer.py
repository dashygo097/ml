from typing import Dict

from termcolor import colored

from ..base_trainer import GNNTrainer, TrainArgs
from .classify import GNNClassifier


class GNNClassifyTrainer(GNNTrainer):
    def __init__(
        self,
        model: GNNClassifier,
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

    def step(self, batch) -> Dict:
        self.optimizer.zero_grad()

        out = self.model(batch.to(self.device))
        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])

        loss.backward()
        self.optimizer.step()

        return {"loss": loss}

    def validate(self) -> None:
        acc = 0.0
        for data in self.valid_data_loader:
            pred = self.model.predict(data)
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                correct = (pred[mask] == data.y[mask]).sum()
                acc = int(correct) / int(mask.sum())

        self.logger.log("valid", {"accuracy": acc}, index=self.n_epochs)
        print(
            f"(Validation {self.n_epochs}) "
            + f": {colored('accuracy', 'green')}: {acc:.4f}"
        )
