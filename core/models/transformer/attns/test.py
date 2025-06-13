import torch
from torch import nn

from ml.core import GroupedQueryAttn

attn = GroupedQueryAttn(128, 8, 4, d_model=64)
inputs = torch.randn(10, 20, 128)
print(attn(inputs).shape)
