from typing import Tuple

import torch
from torch import nn

from ..components import PatchEmbedding


class ViT(nn.Module):
    def __init__(
        self,
        res: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        num_classes: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.res = res
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.embedding = PatchEmbedding(
            self.res, self.patch_size, self.in_channels, self.embed_size
        )
