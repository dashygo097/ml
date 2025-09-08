import torch
import torch.nn.functional as F
from torch import nn


class UnetConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels=None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class UnetDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            UnetConv(in_channels, out_channels),
        )

    @torch.compile
    def forward(self, x):
        return self.down(x)


class UnetUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = UnetConv(in_channels, out_channels, mid_channels=in_channels // 2)

    @torch.compile
    def forward(self, x_0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dim_diff_1 = (x_0.shape[-1] - x.shape[-1]) // 2
        dim_diff_2 = (x_0.shape[-2] - x.shape[-2]) // 2
        pad = (
            dim_diff_1,
            x_0.shape[-1] - x.shape[-1] - dim_diff_1,
            dim_diff_2,
            x_0.shape[-2] - x.shape[-2] - dim_diff_2,
        )
        x = F.pad(x, pad)
        x = torch.concat((x_0, x), dim=1)

        return self.conv(x)
