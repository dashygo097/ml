from typing import Dict

import torch
from torch import nn
from torch_geometric.data import Data

from ..base_trainer import GNNTrainArgs, GNNTrainer


class RecommendNegativeSampler:
    def __init__(self, user_items_dict, n_items) -> None:
        self.user_items_dict = user_items_dict
        self.n_items = n_items

    def sample_negatives(self, user_ids):
        neg_items = []
        for user in user_ids:
            seen = self.user_items_dict[user.item()]
            while True:
                neg = torch.randint(0, self.n_items, (1,)).item()
                if neg not in seen:
                    neg_items.append(neg)
                    break
        return torch.tensor(neg_items)


class ScoreBasedRecommendTrainArgs(GNNTrainArgs):
    def __init__(self, path_or_dict: str | Dict) -> None:
        super().__init__(path_or_dict)


class ScoreBasedRecommendTrainer(GNNTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion,
        user_item_dict,
        args: GNNTrainArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )
        self.negative_sampler = RecommendNegativeSampler(
            user_items_dict=user_item_dict,
            n_items=len(user_item_dict.items()),
        )

    def step(self, batch: Data) -> Dict:
        user_emb, item_emb = self.model.encode(batch)
        user_ids = batch.edge_index[0]

        pos_item_ids = batch.edge_index[1] - self.model.n_users
        neg_item_ids = self.negative_sampler.sample_negatives(user_ids)

        print(user_ids, pos_item_ids, neg_item_ids)

        u_emb = user_emb[user_ids]
        pos_i_emb = item_emb[pos_item_ids]
        neg_i_emb = item_emb[neg_item_ids]

        loss = self.criterion(u_emb, pos_i_emb, neg_i_emb)

        return {"loss": loss}

    def validate(self) -> None: ...
