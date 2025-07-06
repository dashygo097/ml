from typing import Dict

from torch import nn

from ..base_trainer import GNNTrainer, TrainArgs


class GNNClassifyTrainer(GNNTrainer):
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

    def set_dataset(self, dataset) -> None:
        class BaseIterator:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return 1

            def __iter__(self):
                while True:
                    yield self.data
                    break

        self.data_loader = BaseIterator(dataset)

    def step(self, batch) -> Dict:
        self.optimizer.zero_grad()

        out = self.model(batch.to(self.device))
        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])

        loss.backward()
        self.optimizer.step()

        return {"loss": loss}

    def validate(self) -> None: ...
