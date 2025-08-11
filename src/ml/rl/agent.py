import torch
from torch import nn


class RLAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
