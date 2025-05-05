from typing import Optional

import torch
from torch import nn
from torch.nn.utils import prune

from .tracer import Tracer


class Pruner(Tracer):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def parse(self, folder: str = "edited", module_name: Optional[str] = None) -> None:
        if module_name is None:
            self.graph.to_folder(folder, "Pruned" + self.model.__class__.__name__)
        else:
            self.graph.to_folder(folder, module_name)

    def prune_typed(self, pruned_type: type, amount: float = 0.2, n: int = 2) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, pruned_type):
                prune.ln_structured(module, name="weight", amount=amount, n=n, dim=0)
                prune.remove(module, "weight")

        self.graph.recompile()


def shrink_linear_layer(layer: nn.Linear, dim=0):
    mask = layer.weight.detach().abs().sum(dim=1 - dim) != 0
    idx = mask.nonzero(as_tuple=True)[0]

    in_features = layer.in_features if dim == 0 else len(idx)
    out_features = len(idx) if dim == 0 else layer.out_features
    new_layer = nn.Linear(in_features, out_features, bias=layer.bias is not None)

    with torch.no_grad():
        if dim == 0:
            new_layer.weight.copy_(layer.weight[idx])
            if layer.bias is not None:
                new_layer.bias.copy_(layer.bias[idx])
        else:
            new_layer.weight.copy_(layer.weight[:, idx])
            if layer.bias is not None:
                new_layer.bias.copy_(layer.bias)

    return new_layer
