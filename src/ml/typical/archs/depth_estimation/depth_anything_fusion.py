from typing import List, Optional

import torch
from torch import nn


class DepthAnythingPreActResidualLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True
        )

        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        x += residual
        return x


class DepthAnythingFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
        self.res1 = DepthAnythingPreActResidualLayer(hidden_dim)
        self.res2 = DepthAnythingPreActResidualLayer(hidden_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        size=None,
    ) -> torch.Tensor:
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual,
                    size=(hidden_state.shape[2], hidden_state.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            hidden_state += self.res1(residual)

        hidden_state = self.res2(hidden_state)

        modifier = {"scale_factor": 2} if size is None else {"size": size}

        hidden_state = nn.functional.interpolate(
            hidden_state,
            **modifier,
            mode="bilinear",
            align_corners=False,
        )

        return self.proj(hidden_state)


class DepthAnythingFusionStage(nn.Module):
    def __init__(self, hidden_dims: List[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.layers.append(DepthAnythingFusionLayer(hidden_dim))

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        size=None,
    ) -> List[torch.Tensor]:
        hidden_states = hidden_states[::-1]
        fused_hidden_states = []
        fused_hidden_state = None

        for idx, (hidden_state, layer) in enumerate(zip(hidden_states, self.layers)):
            size = (
                hidden_states[idx + 1].shape[2:]
                if idx != (len(hidden_states) - 1)
                else None
            )
            if fused_hidden_state is None:
                fused_hidden_state = layer(hidden_state, size=size)
            else:
                fused_hidden_state = layer(fused_hidden_state, hidden_state, size=size)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states
