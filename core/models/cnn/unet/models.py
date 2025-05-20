import torch
from torch import nn

from .parts import UnetConv, UnetDownBlock, UnetUpBlock


class UnetCNN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        out_channels: int,
        unet_channels: int,
        depth: int,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.unet_channels = unet_channels
        self.depth = depth

        self.in_conv = UnetConv(n_channels, unet_channels)

        self.down_sampler = nn.ModuleList()
        self.up_sampler = nn.ModuleList()

        self.out_conv = nn.Conv2d(unet_channels, out_channels, kernel_size=1)

        for i in range(depth):
            in_channels = unet_channels * 2**i
            out_channels = in_channels * 2
            self.down_sampler.add_module(
                "downsampler_" + str(i),
                UnetDownBlock(
                    in_channels,
                    out_channels,
                ),
            )

            in_channels = unet_channels * 2 ** (depth - i)
            out_channels = in_channels // 2
            self.up_sampler.add_module(
                "upsampler_" + str(i),
                UnetUpBlock(
                    in_channels,
                    out_channels,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x_ram = [x]

        for blk in self.down_sampler:
            x = blk(x)
            x_ram.append(x)

        for i, blk in enumerate(self.up_sampler):
            x = blk(x_ram[self.depth - i - 1], x)

        return self.out_conv(x)
