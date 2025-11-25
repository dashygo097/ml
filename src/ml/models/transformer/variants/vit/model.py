from typing import List, Optional, Tuple

import torch
from torch import nn

from ...components import PatchEmbedding
from ...encoder import EncoderBlock


class ViTBackbone(nn.Module):
    def __init__(
        self,
        embed_size: int,
        patch_size: int,
        n_heads: int,
        n_layers: int,
        res: Tuple[int, int],
        in_channels: int,
        d_inner: Optional[int] = None,
        d_model: Optional[int] = None,
        dropout: float = 0.0,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        # Model parameters
        self.embed_size = embed_size
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_model = d_model if d_model is not None else embed_size
        self.d_inner = d_inner if d_inner is not None else 4 * self.d_model
        self.bias = True
        self.dropout = dropout

        # Image parameters
        self.res = res
        self.in_channels = in_channels

        self.embedding = PatchEmbedding(
            embed_size,
            res,
            patch_size,
            in_channels,
            use_cls_token=use_cls_token,
            dropout=dropout,
        )

        module_list = []
        for _ in range(n_layers):
            module_list.extend(
                [
                    EncoderBlock(
                        embed_size,
                        n_heads,
                        self.d_inner,
                        d_model=self.d_model,
                        norm1=nn.LayerNorm(embed_size, eps=1e-12),
                        norm2=nn.LayerNorm(embed_size, eps=1e-12),
                        bias=True,
                        enable_rope=False,
                        postnorm=False,
                        dropout=dropout,
                    )
                ]
            )

        self.encoder = nn.Sequential(*module_list)
        self.post_norm = nn.LayerNorm(self.d_model, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        return self.post_norm(x)

    def forward_with_intermediates(
        self, x: torch.Tensor, idx: List[int]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.embedding(x)
        intermediates = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in idx:
                intermediates.append(x)
        x = self.post_norm(x)
        return x, intermediates
