from typing import List, Optional

import torch
from torch import nn


class DepthAnythingReassembler(nn.Module):
    def __init__(self, hidden_dim: int, channels: int, factor: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        if factor > 1:
            self.upsample = nn.ConvTranspose2d(
                channels, channels, kernel_size=factor, stride=factor
            )
        elif factor == 1:
            self.upsample = nn.Identity()
        elif factor < 1:
            self.upsample = nn.Conv2d(
                channels, channels, kernel_size=3, stride=int(1 / factor), padding=1
            )


class DepthAnythingReassembleStage(nn.Module):
    def __init__(self, hidden_dims: List[int], reassemble_factors: List[int]) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for channels, factor in zip(hidden_dims, reassemble_factors):
            self.layers.append(
                DepthAnythingReassembler(channels, channels=channels, factor=factor)
            )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        patch_height: Optional[int] = None,
        patch_width: Optional[int] = None,
    ) -> List[torch.Tensor]:
        outs = []

        for idx, hidden_state in enumerate(hidden_states):
            hidden_state = hidden_state[:, 1:]
            batch_size, _, num_channels = hidden_state.shape
            hidden_state = (
                hidden_state.reshape(
                    batch_size, patch_height, patch_width, num_channels
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            hidden_state = self.layers[idx](hidden_state)
            outs.append(hidden_state)

        return outs
