from typing import Callable, List

import torch
from torch import nn


class ViTCNNBasedChangeDetectionHead(nn.Module):
    def __init__(
        self,
        features: int | List[int],
        kernel_sizes: int | List[int],
        num_classes: int,
        patch_size: int,
        act: Callable = nn.Identity(),
        out_act: Callable = nn.Identity(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size: int = patch_size

        init_channels: int = features[0] if isinstance(features, List) else features
        self.decoder = self._build_progressive_decoder(
            init_channels, features, kernel_sizes, num_classes, act, dropout
        )

        branch_layers = []
        diff_layers = []
        branch_layers.extend(
            [
                nn.Conv2d(init_channels, init_channels // 2, 3, padding=1),
                nn.BatchNorm2d(init_channels // 2),
                act,
            ]
        )
        diff_layers.extend(
            [
                nn.Conv2d(init_channels // 2, init_channels, 3, padding=1),
                nn.BatchNorm2d(init_channels),
                act,
            ]
        )

        self.branch_conv = nn.Sequential(*branch_layers)
        self.diff_conv = nn.Sequential(*diff_layers)
        self.out_act = out_act

    def _build_progressive_decoder(
        self,
        init_channels: int,
        features: int | List[int],
        kernel_sizes: int | List[int],
        num_classes: int,
        act: Callable,
        dropout: float,
    ):
        if isinstance(features, List):
            decoder_features = [init_channels] + features + [num_classes]
        else:
            decoder_features = [init_channels] + [features] * 3 + [num_classes]

        if isinstance(kernel_sizes, List):
            decoder_kernels = kernel_sizes + [3] * (
                len(decoder_features) - len(kernel_sizes) - 1
            )
        else:
            decoder_kernels = [kernel_sizes] * (len(decoder_features) - 1)

        layers = []
        current_scale = 1

        for i in range(len(decoder_features) - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        decoder_features[i],
                        decoder_features[i + 1],
                        decoder_kernels[i],
                        padding=decoder_kernels[i] // 2,
                    ),
                    nn.BatchNorm2d(decoder_features[i + 1]),
                    act,
                    nn.Dropout2d(dropout),
                ]
            )

            if (
                current_scale < self.patch_size
                and decoder_features[i + 1] != num_classes
            ):
                scale_factor = 2
                layers.append(
                    nn.Upsample(
                        scale_factor=scale_factor, mode="bilinear", align_corners=False
                    )
                )
                current_scale *= scale_factor

        if current_scale < self.patch_size:
            final_scale = self.patch_size // current_scale
            layers.append(
                nn.Upsample(
                    scale_factor=final_scale, mode="bilinear", align_corners=False
                )
            )

        return nn.Sequential(*layers)

    def _reshape_vit_features(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int((N - 1) ** 0.5)

        x = x[:, 1:].transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)

        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1_spatial = self._reshape_vit_features(x1)
        x2_spatial = self._reshape_vit_features(x2)

        feat1_processed = self.branch_conv(x1_spatial)
        feat2_processed = self.branch_conv(x2_spatial)

        feature_diff = torch.abs(feat1_processed - feat2_processed)
        feature_map = self.diff_conv(feature_diff)

        output = self.decoder(feature_map)

        return self.out_act(output)


class ViTFPNBasedChangeDetection(nn.Module):
    def __init__(
        self,
        features: int,
        num_classes: int,
        patch_size: int,
        act: Callable = nn.Identity(),
        out_act: Callable = nn.Identity(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

    def forward(
        self, x1: List[torch.Tensor], x2: List[torch.Tensor]
    ) -> torch.Tensor: ...
