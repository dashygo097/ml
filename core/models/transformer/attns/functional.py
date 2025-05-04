import math
from typing import Tuple

import torch
import torch.nn.functional as F


def masked_softmax(
    x: torch.Tensor,
    masked=None,
) -> torch.Tensor:
    x = x - torch.max(x)

    if masked is None:
        Y = F.softmax(x, dim=-1)
    elif masked == "^":
        for i in range(x.shape[-2]):
            x[..., i, i + 1 : :] = -1e7
        Y = F.softmax(x, dim=-1)
    elif isinstance(masked, torch.Tensor):
        x = x.masked_fill(masked == 0, float("-inf"))
        return F.softmax(x, dim=-1)
    else:
        raise ValueError("Invalid masked value")

    return Y


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    masked=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = Q.shape[-1]
    results = (Q @ K.transpose(-1, -2)) / math.sqrt(d)
    weights = masked_softmax(results, masked=masked)
    outputs = weights @ V

    return outputs, weights
