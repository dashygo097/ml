from typing import Tuple

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(
        self, res: Tuple[int, int], patch_size: int, in_channels: int, embed_size: int
    ) -> None:
        super().__init__()
        self.res = res
        self.patch_size = patch_size
        self.num_patches = res[0] * res[1] // patch_size
        self.in_channels = in_channels
        self.d_model = embed_size
        self.embed_size = embed_size

        self.proj = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.embed_size,
                kernel_size=self.patch_size,
                padding=patch_size,
            ),
            Rearrange("B C H W -> B (H W) C"),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embed_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.res[0] and W == self.res[1], (
            f"[ERROR] img_size mismatch ({H}, {W}) != ({self.res[0]}, {self.res[1]})"
        )

        x = self.proj(x)
        cls_tokens = repeat(self.cls_token, "1 1 d -> B 1 d", B=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding

        return x
