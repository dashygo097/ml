from typing import List, Tuple

import torch
from termcolor import colored
from torch import nn
from torch_geometric.data import Data

from ..base import GNNEncoder
from .heads import ScoreBasedRecommendHead


class ScoreBasedRecommender(nn.Module):
    def __init__(self, encoder: GNNEncoder, n_users: int, n_items: int):
        super().__init__()
        if encoder.in_features is None:
            raise ValueError(
                colored(
                    "[ERROR] Encoder must have in_features defined.",
                    color="red",
                    attrs=["bold"],
                )
            )
        self.n_users = n_users
        self.n_items = n_items

        self.user_embedding = nn.Embedding(n_users, encoder.in_features)
        self.item_embedding = nn.Embedding(n_items, encoder.in_features)
        self.encoder = encoder
        self.head = ScoreBasedRecommendHead()

    def forward(self, data: Data) -> Tuple[torch.Tensor, ...]:
        embeddings = []
        x = torch.cat(
            [
                self.user_embedding.weight,
                self.item_embedding.weight,
            ],
            dim=0,
        )
        embeddings.append(x)

        data.x = x
        embeddings.extend(self.encoder.feats(data))

        final_emb = torch.mean(torch.stack(embeddings, dim=0), dim=0)
        user_emb, item_emb = (
            final_emb[: self.user_embedding.num_embeddings],
            final_emb[self.user_embedding.num_embeddings :],
        )
        return user_emb, item_emb

    def recommmend(self, user_id: int, data: Data, topk: int = 1) -> List[int]:
        if user_id < 0 or user_id >= self.n_users:
            raise ValueError(
                colored(
                    "[ERROR] User ID must be within the range of the number of users.",
                    color="red",
                    attrs=["bold"],
                )
            )

        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward(data)
            return self.head(user_id, user_emb, item_emb, topk)
