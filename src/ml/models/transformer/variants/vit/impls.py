import torch
from typing import Tuple
from torch import nn
from .model import ViTBackbone, ViTConfig

from ....heads.change_detection import ViTCNNBasedChangeDetectionHead
from ....heads.classify import ClassifyHead
from ....heads.obj_detection import DeTRThetaBasedOBBDetectionHead, ObjDetecttionHead2D

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
        self.head = ClassifyHead(config.embed_size, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x)[:, 0, :])


class ViTObjDetector2D(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
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
        ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

class ViTDeTRThetaBasedOBBDetector(nn.Module):
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
        self.head = DeTRThetaBasedOBBDetectionHead(
            embed_size=config.embed_size,
            num_classes=config.num_classes,
            num_queries=config.head_num_queries,
            n_heads=config.head_n_heads,
            n_layers=config.head_n_layers,
            d_model=config.head_d_model,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.vit(x)
        return self.head(x)


class ViTCNNBasedChangeDetector(nn.Module):
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

        self.head = ViTCNNBasedChangeDetectionHead(
            features=[config.embed_size] + config.head_hidden_features,
            kernel_sizes=config.head_kernel_sizes,
            num_classes=config.num_classes,
            patch_size=config.patch_size,
            act=nn.GELU(),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([x1, x2], dim=0)
        x1, x2 = self.vit(x).chunk(2, dim=0)
        return self.head(x1, x2)


