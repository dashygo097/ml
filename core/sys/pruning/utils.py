import torch
from torch import nn


def has_multi_dim(module: nn.Module) -> bool:
    if hasattr(module, "weight"):
        return module.weight.data.ndim > 1

    else:
        return False


def should_pass(module: nn.Module) -> bool:
    if has_multi_dim(module):
        return True
    else:
        return False
