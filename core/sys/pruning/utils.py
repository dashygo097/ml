from torch import nn


def has_multi_dim(module: nn.Module) -> bool:
    if not hasattr(module, "weight"):
        return False
    else:
        return module.weight.data.ndim > 1
