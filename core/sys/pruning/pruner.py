from typing import Optional

import torch
from torch import nn
from torch.nn.utils import prune

from ..tracer import Tracer


class Pruner(Tracer):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def parse(self, folder: str = "edited", module_name: Optional[str] = None) -> None:
        if module_name is None:
            self.graph.to_folder(folder, "Pruned" + self.model.__class__.__name__)
        else:
            self.graph.to_folder(folder, module_name)

    def prune_typed(
        self, pruned_type: type, amount: float = 0.2, n: int = 2
    ) -> None: ...
