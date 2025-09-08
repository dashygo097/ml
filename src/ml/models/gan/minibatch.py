import torch
from torch import nn


class MiniBatch1d(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        inner_dim: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.inner_dim = inner_dim

        self.T = nn.Parameter(
            torch.randn(in_features, out_features, inner_dim, requires_grad=True) * 0.1
        )

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        x_res = x

        x = x @ self.T.view(self.in_features, self.out_features * self.inner_dim)
        x = x.view(B, self.out_features, self.inner_dim)
        x = (x.unsqueeze(0) - x.unsqueeze(1)).abs().sum(dim=-1)

        # mask = ~torch.eye(B).bool()
        # mask = mask.unsqueeze(-1).expand(B, B, self.out_features)

        # x = torch.exp(-x) * mask
        # x = x.sum(dim=1) / (B - 1)

        x = torch.exp(-x).mean(dim=1)

        return torch.cat([x_res, x], dim=-1)


class MiniBatchHead1d(nn.Module):
    def __init__(self, in_features: int, out_features: int, inner_dim: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.inner_dim = inner_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
