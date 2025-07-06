from typing import Dict

import torch

from ..base_trainer import GNNTrainer, TrainArgs
from .encoder import GraphMAE


class GraphMAETrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict) -> None:
        super().__init__(path_or_dict)
        self.mask_ratio = self.args.get("mask_ratio", 0.5)


class GraphMAEPretrainer(GNNTrainer):
    def __init__(
        self,
        model: GraphMAE,
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

        num_nodes = batch.x.size(0)
        mask = torch.rand(num_nodes, device=batch.x.device) < self.args.mask_ratio

        x_rec, x_target = self.model.reconstruct(
            batch.x.clone(), batch.edge_index, mask
        )

        loss = self.criterion(x_rec, x_target)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss}
