from typing import List, Tuple

import torch
from termcolor import colored
from torch import nn
from torch_geometric.data import Data

from ..base import GNNEncoder
from .heads import ScoreBasedRecommendHead


class ScoreBasedRecommender(nn.Module):
    def __init__(self, encoder: GNNEncoder, num_users: int, num_items: int):
        super().__init__()
        if encoder.in_features is None:
            raise ValueError(
                colored(
                    "[ERROR] Encoder must have in_features defined.",
                    color="red",
                    attrs=["bold"],
                )
            )
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding = nn.Embedding(num_users, encoder.in_features)
        self.item_embedding = nn.Embedding(num_items, encoder.in_features)
        self.encoder = encoder
        self.head = ScoreBasedRecommendHead()

    def recommend(self, user_id: int, data: Data, topk: int = 1) -> List[int]:
        self.eval()
        with torch.no_grad():
            return self.forward(user_id, data, topk)

    def forward(self, user_id: int, data: Data, topk: int = 1) -> List[int]:
        if user_id < 0 or user_id >= self.num_users:
            raise ValueError(
                colored(
                    "[ERROR] User ID must be within the range of the number of users.",
                    color="red",
                    attrs=["bold"],
                )
            )

        user_emb, item_emb = self.encode(data)
        return self.decoder(user_id, user_emb, item_emb, topk)

    def decoder(
        self, user_id: int, user_emb: torch.Tensor, item_emb: torch.Tensor, topk: int
    ) -> List[int]:
        if user_id < 0 or user_id >= self.num_users:
            raise ValueError(
                colored(
                    "[ERROR] User ID must be within the range of the number of users.",
                    color="red",
                    attrs=["bold"],
                )
            )
        return self.head(user_id, user_emb, item_emb, topk)

    def encode(self, data: Data) -> Tuple[torch.Tensor, ...]:
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
