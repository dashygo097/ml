from typing import Optional

from torch import nn


class PrintShape(nn.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name

    def forward(self, x):
        if self.name:
            print(f"{self.name} shape: {x.shape}")
        else:
            print(f"Shape: {x.shape}")
        return x
