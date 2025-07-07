from typing import Optional

import torch
import torch_geometric.nn as gnn
from torch import nn


class Node2Vec(nn.Module):
    def __init__(
        self,
        edge_index: torch.Tensor,
        embed_size: int = 128,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 10,
        num_negative_samples: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        sparse: bool = True,
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.embed_size = embed_size
        self.pretrained_path = pretrained_path
        self.freeze = freeze

        self.emb_model = gnn.Node2Vec(
            edge_index=edge_index,
            embedding_dim=embed_size,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=sparse,
        ).to(device)

        if pretrained_path:
            self.load(pretrained_path)
            if freeze:
                for param in self.emb_model.parameters():
                    param.requires_grad = False

    def forward(self) -> torch.Tensor:
        return self.emb_model.embedding.weight

    def train_node2vec(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 128,
        log_every: int = 10,
    ):
        loader = self.node2vec.loader(batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SparseAdam(self.node2vec.parameters(), lr=lr)

        self.node2vec.train()
        for epoch in range(epochs):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.node2vec.loss(
                    pos_rw.to(self.device), neg_rw.to(self.device)
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def save(self, path: str):
        torch.save(self.emb_model.embedding.weight.data.cpu(), path)

    def load(self, path: str):
        self.emb_model.embedding.weight.data.copy_(
            torch.load(path, map_location=self.device)
        )
