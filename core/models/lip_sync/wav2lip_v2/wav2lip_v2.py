import torch
from torch import nn


class Wav2Lip_v2(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()

        self.device = device
