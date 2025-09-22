from typing import Optional, Tuple

import torch
from torch import nn

from ..transformer import DecoderBlock


class DeTRThetaBasedOBBDetectionHead(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_classes: int,
        num_queries: int,
        n_heads: int,
        n_layers: int = 6,
        d_model: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_size: int = embed_size
        self.num_classes: int = num_classes
        self.num_queries: int = num_queries
        self.d_model: int = d_model if d_model is not None else embed_size
        self.n_heads: int = n_heads
        self.dropout: float = dropout

        self.query_embed = nn.Embedding(num_queries, self.d_model)
        self.input_proj = nn.Linear(embed_size, self.d_model)

        decoder_layers = []
        for _ in range(n_layers):
            decoder_layers.extend(
                [
                    DecoderBlock(
                        embed_size=self.d_model,
                        n_heads=n_heads,
                        d_model=self.d_model,
                        d_inner=self.d_model * 4,
                        norm1=nn.LayerNorm(self.d_model, eps=1e-12),
                        norm2=nn.LayerNorm(self.d_model, eps=1e-12),
                        norm3=nn.LayerNorm(self.d_model, eps=1e-12),
                        enable_rope=False,
                        bias=True,
                        postnorm=False,
                        dropout=dropout,
                    )
                ]
            )

        self.decoder = nn.ModuleList(decoder_layers)
        self.cls_head = nn.Linear(self.d_model, num_classes + 1)
        self.bbox_head = nn.Linear(self.d_model, 4)
        self.angle_head = nn.Linear(self.d_model, 1)

    def forward(
        self, vit_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, E = vit_features.shape
        memory = self.input_proj(vit_features)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        for layer in self.decoder:
            decoder_output = layer(queries, memory, is_causal=False)

        cls_logits = self.cls_head(decoder_output)
        bbox_preds = self.bbox_head(decoder_output).sigmoid()
        angle_preds = self.angle_head(decoder_output).sigmoid()
        return cls_logits, bbox_preds, angle_preds
