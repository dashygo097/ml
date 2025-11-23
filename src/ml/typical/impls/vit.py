from typing import Tuple

import torch
from torch import nn

from ...models import ViTBackbone, ViTConfig
from ..heads import *


class ViTRawModel(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.vit = ViTBackbone(
            embed_size=config.embed_size,
            patch_size=config.patch_size,
            n_heads=config.n_heads,
            n_layers=config.num_layers,
            res=config.res,
            in_channels=config.in_channels,
            d_inner=config.d_inner,
            d_model=config.d_model,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


class ViTClassifier(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.vit = ViTBackbone(
            embed_size=config.embed_size,
            patch_size=config.patch_size,
            n_heads=config.n_heads,
            n_layers=config.num_layers,
            res=config.res,
            in_channels=config.in_channels,
            d_inner=config.d_inner,
            d_model=config.d_model,
            dropout=config.dropout,
        )
        if config.head_type == "mlp":
            self.head = ClassifyHead(
                [config.embed_size] + self.config.head_hidden_features,
                config.num_classes,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unsupported head type: {config.head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x)[:, 0, :])


class ViTChangeDetector(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.vit = ViTBackbone(
            embed_size=config.embed_size,
            patch_size=config.patch_size,
            n_heads=config.n_heads,
            n_layers=config.num_layers,
            res=config.res,
            in_channels=config.in_channels,
            d_inner=config.d_inner,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        if config.head_type == "cnn":
            self.head = CNNBasedChangeDetectionHead(
                features=[config.embed_size] + config.head_hidden_features,
                kernel_sizes=config.head_kernel_sizes,
                num_classes=config.num_classes,
                patch_size=config.patch_size,
                act=nn.GELU(),
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unsupported head type: {config.head_type}")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([x1, x2], dim=0)
        x1, x2 = self.vit(x).chunk(2, dim=0)
        return self.head(x1, x2)
