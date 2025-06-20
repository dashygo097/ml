import torch
from typing import List
from .conv import Conv2d, Conv2dTranspose
from torch import nn


class FaceDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        1024, 512, kernel_size=3, stride=1, padding=0
                    ),  # 3,3
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 6, 6
                nn.Sequential(
                    Conv2dTranspose(
                        768, 384, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 12, 12
                nn.Sequential(
                    Conv2dTranspose(
                        512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 24, 24
                nn.Sequential(
                    Conv2dTranspose(
                        320, 128, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 48, 48
                nn.Sequential(
                    Conv2dTranspose(
                        160, 64, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),
            ]
        )  # 96,96

    def forward(
        self, audio_embedding: torch.Tensor, feats: List[torch.Tensor]
    ) -> torch.Tensor:
        x = audio_embedding
        for f in self.blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()
        return x
