from typing import Optional

from torch import nn


def has_in_attr(module: nn.Module) -> Optional[str]:
    if hasattr(module, "in_features"):
        return "in_features"
    elif hasattr(module, "in_channels"):
        return "in_channels"
    else:
        return None


def has_independent_in_and_out_attr(module: nn.Module) -> Optional[str]:
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return "yes"

    elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
        return "yes"

    else:
        return None


def has_out_attr(module: nn.Module) -> Optional[str]:
    if hasattr(module, "out_features"):
        return "out_features"
    elif hasattr(module, "out_channels"):
        return "out_channels"
    else:
        return None
