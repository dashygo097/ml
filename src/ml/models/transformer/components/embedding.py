from typing import Optional, Tuple

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
        use_mask_token: bool = False,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.patch_size = patch_size
        self.res = res
        self.in_channels = in_channels
        self.use_cls_token = use_cls_token
        self.use_mask_token = use_mask_token

        self.base_grid_h = res[0] // patch_size
        self.base_grid_w = res[1] // patch_size
        self.base_num_patches = self.base_grid_h * self.base_grid_w

        self.proj = nn.Conv2d(
            in_channels,
            embed_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
            nn.init.normal_(self.cls_token, std=0.02)
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, embed_size))
            nn.init.normal_(self.mask_token, std=0.02)

        num_pos = self.base_num_patches + (1 if use_cls_token else 0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_pos, embed_size))
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def _get_pos_embed(
        self,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        if grid_h == self.base_grid_h and grid_w == self.base_grid_w:
            return self.pos_embedding

        if self.use_cls_token:
            cls_pos_embed = self.pos_embedding[:, :1, :]
            patch_pos_embed = self.pos_embedding[:, 1:, :]
        else:
            cls_pos_embed = None
            patch_pos_embed = self.pos_embedding

        patch_pos_embed = patch_pos_embed.reshape(
            1, self.base_grid_h, self.base_grid_w, self.embed_size
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, embed_dim, h, w)

        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, h, w, embed_dim)
        patch_pos_embed = patch_pos_embed.reshape(1, grid_h * grid_w, self.embed_size)

        if self.use_cls_token and cls_pos_embed is not None:
            patch_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)

        return patch_pos_embed

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        x = self.proj(x)  # (B, embed_dim, grid_h, grid_w)
        _, _, grid_h, grid_w = x.shape

        x = x.flatten(2).transpose(1, 2)  # (B, grid_h*grid_w, embed_dim)

        if self.use_mask_token and mask is not None:
            x = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.unsqueeze(0).unsqueeze(0),
                x,
            )

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + grid_h*grid_w, embed_dim)

        pos_embed = self._get_pos_embed(grid_h, grid_w)
        x = x + pos_embed

        x = self.dropout(x)
        return x
