from typing import Callable, List, Tuple

import torch
from torch import nn


class ViTCNNBasedChangeDetectionHead(nn.Module):
    def __init__(
        self,
        features: int | List[int],
        kernel_sizes: int | List[int],
        num_classes: int,
        patch_size: int,
        forward_type: str = "subtract",
        act: Callable = nn.Identity(),
        out_act: Callable = nn.Identity(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.forward_type = forward_type

        cnn_features = (
            features + [num_classes]
            if isinstance(features, List)
            else [features] + [num_classes]
        )
        cnn_kernels = (
            kernel_sizes
            if isinstance(kernel_sizes, List)
            else [kernel_sizes] * (len(cnn_features) - 1)
        )

        decoder_layers = []
        for i in range(len(cnn_features) - 1):
            decoder_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=cnn_features[i],
                        out_channels=cnn_features[i + 1],
                        kernel_size=cnn_kernels[i],
                        padding=cnn_kernels[i] // 2,
                    ),
                    act,
                    nn.Dropout2d(dropout),
                ]
            )
        decoder_layers.append(
            nn.Conv2d(
                in_channels=cnn_features[-1],
                out_channels=num_classes,
                kernel_size=1,
            )
        )
        decoder_layers.append(
            nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False)
        )

        self.decoder = nn.Sequential(*decoder_layers)
        self.out_act = out_act

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, E = x1.shape
        H = W = int((C - 1) ** 0.5)
        x1 = x1.transpose(1, 2).contiguous()
        x2 = x2.transpose(1, 2).contiguous()

        if self.forward_type == "subtract":
            feature_map = (x1 - x2)[..., 1:].view(B, E, H, W)
        elif self.forward_type == "concat":
            feature_map = torch.cat((x1, x2), dim=-1)[..., 1:].view(B, 2 * E, H, W)
        else:
            raise ValueError(f"Unknown forward_type: {self.forward_type}")

        return self.out_act(self.decoder(feature_map))
