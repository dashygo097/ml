from typing import Dict

from torch import nn

from ..base_trainer import GNNTrainer, TrainArgs


class GNNEmbeddingTrainer(GNNTrainer):
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
        self.data_loader = self.model.emb_model.loader(
            batch_size=self.args.batch_size, shuffle=self.args.is_shuffle
        )

    def step(self, batch) -> Dict:
        loader = self.model.emb_model.loader(
            batch_size=self.args.batch_size, shuffle=self.args.is_shuffle
        )

        total_loss = 0
        for pos_rw, neg_rw in loader:
            self.optimizer.zero_grad()
            loss = self.model.emb_model.loss(
                pos_rw.to(self.device), neg_rw.to(self.device)
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {"loss": total_loss}
