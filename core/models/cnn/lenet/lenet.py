from typing import OrderedDict

import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()

        module_list = []

        module_list.extend(
            [
                ("conv2d_0", nn.Conv2d(1, 6, kernel_size=5, padding=2)),
                ("bn2d_0", nn.BatchNorm2d(6)),
                ("relu_0", nn.ReLU()),
                ("pool2d_0", nn.AvgPool2d(kernel_size=2, stride=2)),
                ("relu_1", nn.ReLU()),
                ("conv2d_1", nn.Conv2d(6, 16, kernel_size=5)),
                ("bn2d_1", nn.BatchNorm2d(16)),
                ("pool2d_1", nn.AvgPool2d(kernel_size=2, stride=2)),
                ("flatten", nn.Flatten()),
                ("linear_0", nn.Linear(16 * 5 * 5, 120)),
                ("bn1d_0", nn.BatchNorm1d(120)),
                ("relu_2", nn.ReLU()),
                ("linear_1", nn.Linear(120, 84)),
                ("bn1d_1", nn.BatchNorm1d(84)),
                ("relu_3", nn.ReLU()),
                ("linear_2", nn.Linear(84, 10)),
            ]
        )

        self.seq = nn.Sequential(OrderedDict(module_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.seq(x)
