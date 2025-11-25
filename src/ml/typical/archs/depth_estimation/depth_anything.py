from typing import List

import torch
from torch import nn

from .depth_anything_fusion import DepthAnythingFusionStage
from .depth_anythingz_reassambler import DepthAnythingReassembleStage


class DepthAnythingNeck(nn.Module):
    def __init__(
        self,
        neck_hidden_dims: List[int],
        reassemble_factors: List[int],
        fusion_hidden_dim: int,
    ):
        super().__init__()
        self.neck_hidden_dims = neck_hidden_dims
        self.fusion_hidden_dim = fusion_hidden_dim

        self.reassemble_stage = DepthAnythingReassembleStage(
            neck_hidden_dims, reassemble_factors
        )

        self.convs = nn.ModuleList()
        for channel in neck_hidden_dims:
            self.convs.append(
                nn.Conv2d(
                    channel,
                    fusion_hidden_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )

        self.fusion_stage = DepthAnythingFusionStage(neck_hidden_dims)

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        patch_height=None,
        patch_width=None,
    ) -> List[torch.Tensor]:
        if len(hidden_states) != len(self.neck_hidden_dims):
            raise ValueError(
                "The number of hidden states should be equal to the number of neck hidden dims."
            )

        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)
        features = (self.convs[i](feature) for i, feature in enumerate(hidden_states))
        output = self.fusion_stage(features)

        return output


class DepthAnythingDepthEstimationHead(nn.Module):
    def __init__(
        self,
        head_in_index: int,
        patch_size: int,
        fusion_hidden_dim: int,
        head_hidden_dim: int,
        max_depth: float,
        head_type: str,
    ) -> None:
        super().__init__()
        self.head_in_index = head_in_index
        self.patch_size = patch_size
        self.fusion_hidden_dim = fusion_hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.max_depth = max_depth

        self.conv1 = nn.Conv2d(
            fusion_hidden_dim,
            fusion_hidden_dim // 2,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            fusion_hidden_dim // 2,
            head_hidden_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.act1 = nn.ReLU()
        self.conv3 = nn.Conv2d(head_hidden_dim, 1, kernel_size=1)
        if head_type == "relative":
            self.act2 = nn.ReLU()
        elif head_type == "metric":
            self.act2 = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown depth estimation type: {head_type}")

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        patch_height: int,
        patch_width: int,
    ) -> torch.Tensor:
        x = hidden_states[self.head_in_index]
        x = self.conv1(x)
        x = nn.functional.interpolate(
            x,
            size=(patch_height * self.patch_size, patch_width * self.patch_size),
            mode="bilinear",
            align_corners=False,
        )
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        x = self.act2(x) * self.max_depth
        return x
