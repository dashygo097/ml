from collections import deque
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.fx import Node
from torch.nn.utils import prune

from ..tracer import Tracer
from ..utils import has_in_attr, has_independent_in_and_out_attr, has_out_attr


class Pruner(Tracer):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def parse(self, folder: str = "edited", module_name: Optional[str] = None) -> None:
        if module_name is None:
            self.graph.to_folder(folder, "Pruned" + self.model.__class__.__name__)
        else:
            self.graph.to_folder(folder, module_name)

    def prune_typed(self, pruned_type: type, amount: float = 0.2, n: int = 2) -> Dict:
        pruned_output = {}
        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                module = self.graph.get_submodule(node.target)
                if isinstance(module, pruned_type):
                    prune.ln_structured(
                        module, name="weight", amount=amount, n=n, dim=0
                    )

                    mask = module.weight_mask.sum(dim=-1).to(torch.bool)
                    pruned_output[node] = torch.where(mask, 1, 0)

                    prune.remove(module, "weight")
                    self._shrink_linear_node(node, mask, is_prune=True)
                    self.update(node, mask)

        self.graph.recompile()
        return pruned_output

    def _shrink_linear_node(
        self, target: Node, mask: torch.Tensor, is_prune: bool
    ) -> None:
        target_module = self.graph.get_submodule(target.target)
        if isinstance(target_module, nn.Linear):
            leftover = torch.where(mask, 1, 0).sum()

            new_layer = nn.Linear(
                target_module.in_features if is_prune else int(leftover),
                int(leftover) if is_prune else target_module.out_features,
                bias=target_module.bias is not None,
            )

            with torch.no_grad():
                new_layer.weight.data = (
                    target_module.weight.data[mask, :]
                    if is_prune
                    else target_module.weight.data[:, mask]
                )
                if target_module.bias is not None:
                    new_layer.bias.data = target_module.bias.data
                self.replace(target, lambda: new_layer)

    def _shrink_bn_node(self, target: Node, mask: torch.Tensor) -> None:
        target_module = self.graph.get_submodule(target.target)
        if isinstance(target_module, nn.BatchNorm1d):
            leftover = torch.where(mask, 1, 0).sum()
            new_layer = nn.BatchNorm1d(num_features=int(leftover))
            with torch.no_grad():
                new_layer.weight.data = target_module.weight.data[mask]
                new_layer.bias.data = target_module.bias.data[mask]
                self.replace(target, lambda: new_layer)

    def update(self, target: Node, mask: torch.Tensor) -> List[Node]:
        # bfs traversal
        queue = deque(self.fetch_neighbors(target)["output"])
        visited = []
        while queue:
            current_node = queue.popleft()
            visited.append(current_node)

            current_module = self.graph.get_submodule(current_node.target)
            if isinstance(current_module, nn.Linear):
                self._shrink_linear_node(current_node, mask, is_prune=False)
            elif isinstance(current_module, nn.BatchNorm1d):
                self._shrink_bn_node(current_node, mask)

            if has_independent_in_and_out_attr(current_module) is None:
                for n in self.fetch_neighbors(current_node)["output"]:
                    if n not in visited:
                        queue.append(n)

        print(visited)
        return visited
