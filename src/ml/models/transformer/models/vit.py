from typing import Optional, OrderedDict, Tuple

import torch
from torch import nn

from ..components import PatchEmbedding
from ..encoder import EncoderBlock


class ViTBackbone(nn.Module):
    def __init__(
        self,
        embed_size: int,
        patch_size: int,
        n_heads: int,
        num_layers: int,
        res: Tuple[int, int],
        in_channels: int,
        d_inner: Optional[int] = None,
        d_model: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Model parameters
        self.embed_size = embed_size
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.d_model = d_model if d_model is not None else embed_size
        self.d_inner = d_inner if d_inner is not None else 4 * self.d_model
        self.bias = True
        self.dropout = dropout

        # Image parameters
        self.res = res
        self.in_channels = in_channels

        self.embedding = PatchEmbedding(res, patch_size, in_channels, embed_size)

        module_list = []
        for i in range(num_layers):
            module_list.extend(
                [
                    (
                        f"blk_{i}",
                        EncoderBlock(
                            embed_size,
                            n_heads,
                            self.d_inner,
                            d_model=self.d_model,
                            norm=nn.LayerNorm(embed_size, eps=1e-12),
                            bias=True,
                            enable_rope=False,
                            dropout=dropout,
                        ),
                    ),
                ]
            )

        self.encoder = nn.Sequential(OrderedDict(module_list))
        self.post_layernorm = nn.LayerNorm(self.d_model, eps=1e-12)
        self.fc = nn.Linear(768, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        return self.post_layernorm(x)
