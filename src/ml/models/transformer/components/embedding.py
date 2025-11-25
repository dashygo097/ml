from typing import Tuple

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        embed_size: int,
        res: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        dropout: float = 0.0,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.res = res
        self.patch_size = patch_size
        self.num_patches = (res[0] // patch_size) * (res[1] // patch_size)
        self.in_channels = in_channels
        self.d_model = embed_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.use_cls_token = use_cls_token

        self.proj = nn.Conv2d(
            in_channels, self.d_model, kernel_size=patch_size, stride=patch_size
        )
        self.out_dropout = nn.Dropout(dropout)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.d_model)
        )
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding

        return self.out_dropout(x)
