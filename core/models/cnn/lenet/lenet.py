import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, act: nn.Module = nn.Sigmoid()) -> None:
        super(LeNet, self).__init__()
        self.act = act
        self.seq = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
            self.act,
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.BatchNorm1d(120),
            self.act,
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            self.act,
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.seq(x)
