from collections import deque
from typing import Dict, List, Optional, overload

import torch
from torch import nn
from torch.fx import Node
from torch.nn.utils import prune

from ..tracer import Tracer
from .utils import has_multi_dim


class Pruner(Tracer):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def parse(self, folder: str = "edited", module_name: Optional[str] = None) -> None:
        if module_name is None:
            self.graph.to_folder(folder, "Pruned" + self.model.__class__.__name__)
        else:
            self.graph.to_folder(folder, module_name)

    def update(self, target: Node, mask: torch.Tensor) -> List[Node]:
        queue = deque(self.fetch_neighbors(target)["output"])
        visited = []
        while queue:
            current_node = queue.popleft()
            visited.append(current_node)

            self._update_node(current_node, mask)

            current_module = self.graph.get_submodule(current_node.target)
            if not has_multi_dim(current_module):
                for n in self.fetch_neighbors(current_node)["output"]:
                    if n not in visited:
                        queue.append(n)

        return visited

    @overload
    def prune(
        self,
        target: Optional[str] | Optional[Node],
        amount: float = 0.2,
        n: int = 2,
    ) -> Dict:
        return self.prune(target, amount, n)

    @overload
    def prune(self, target: nn.Module, amount: float = 0.2, n: int = 2) -> Dict:
        return self.prune(target, amount, n)

    @overload
    def prune(self, target: type, amount: float = 0.2, n: int = 2) -> Dict:
        return self.prune(target, amount, n)

    def prune(
        self,
        target: Optional[Node] | Optional[str] | nn.Module | type,
        amount: float = 0.2,
        n: int = 2,
    ) -> Dict:
        pruned_output = {}
        if target is None:
            raise ValueError("Target cannot be None.")

        elif isinstance(target, str):
            pruned_output = self.prune_node(target, amount, n)

        elif isinstance(target, Node):
            pruned_output = self.prune_node(target, amount, n)

        elif isinstance(target, nn.Module):
            pruned_output = self.prune_module(target, amount, n)

        elif isinstance(target, type):
            pruned_output = self.prune_typed(target, amount, n)

        return pruned_output

    def prune_node(
        self, target: Optional[Node] | Optional[str], amount: float = 0.2, n: int = 2
    ) -> Dict:
        pruned_output = {}
        target = self.fetch(target)
        if isinstance(target, Node):
            module = self.graph.get_submodule(target.target)

            if isinstance(module, nn.Linear):
                if self.fetch_neighbors(target)["output"]:
                    prune.ln_structured(
                        module, name="weight", amount=amount, n=n, dim=0
                    )
                    prune.remove(module, "weight")
                    weight_importance = torch.norm(module.weight, p=n, dim=1)
                    num_to_keep = int(module.weight.size(0) * (1 - amount))
                    _, indices = torch.topk(weight_importance, num_to_keep)
                    mask = torch.zeros_like(weight_importance, dtype=torch.bool)
                    mask[indices] = True

                    pruned_output[target] = torch.where(mask, 1, 0)

                    self._shrink_linear_node(target, mask, is_prune=True)
                    self.update(target, mask)

        self.graph.recompile()
        return pruned_output

    def prune_module(self, target: nn.Module, amount: float = 0.2, n: int = 2) -> Dict:
        pruned_output = {}

        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                module = self.graph.get_submodule(node.target)
                if module is target:
                    prune.ln_structured(
                        module, name="weight", amount=amount, n=n, dim=0
                    )
                    prune.remove(module, "weight")
                    weight_importance = torch.norm(module.weight, p=n, dim=1)
                    num_to_keep = int(module.weight.size(0) * (1 - amount))
                    _, indices = torch.topk(weight_importance, num_to_keep)
                    mask = torch.zeros_like(weight_importance, dtype=torch.bool)
                    mask[indices] = True

                    pruned_output[node] = torch.where(mask, 1, 0)

                    if isinstance(module, nn.Linear):
                        if self.fetch_neighbors(node)["output"]:
                            self._shrink_linear_node(node, mask, is_prune=True)
                            self.update(node, mask)
        self.graph.recompile()
        return pruned_output

    def prune_typed(self, target: type, amount: float = 0.2, n: int = 2) -> Dict:
        pruned_output = {}

        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                module = self.graph.get_submodule(node.target)
                if isinstance(module, target):
                    prune.ln_structured(
                        module, name="weight", amount=amount, n=n, dim=0
                    )
                    prune.remove(module, "weight")

                    weight_importance = torch.norm(module.weight, p=n, dim=1)
                    num_to_keep = int(module.weight.size(0) * (1 - amount))
                    _, indices = torch.topk(weight_importance, num_to_keep)
                    mask = torch.zeros_like(weight_importance, dtype=torch.bool)
                    mask[indices] = True

                    pruned_output[node] = torch.where(mask, 1, 0)

                    if target == nn.Linear:
                        if self.fetch_neighbors(node)["output"]:
                            self._shrink_linear_node(node, mask, is_prune=True)
                            self.update(node, mask)

        self.graph.recompile()
        return pruned_output

    def _update_node(self, target: Node, mask: torch.Tensor) -> None:
        current_module = self.graph.get_submodule(target.target)
        if isinstance(current_module, nn.Linear):
            self._shrink_linear_node(target, mask, is_prune=False)
        elif isinstance(current_module, nn.BatchNorm1d):
            self._shrink_bn_node(target, mask, is_prune=False)

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
                if is_prune:
                    new_layer.weight.data = target_module.weight.data[mask, :]
                    if target_module.bias is not None:
                        new_layer.bias.data = target_module.bias.data[mask]
                else:
                    scale_factor = target_module.weight.data.shape[1] / mask.sum()
                    new_layer.weight.data = (
                        target_module.weight.data[:, mask] * scale_factor
                    )
                    if target_module.bias is not None:
                        new_layer.bias.data = target_module.bias.data

                self.replace(target, lambda: new_layer)

    def _shrink_bn_node(self, target: Node, mask: torch.Tensor, is_prune: bool) -> None:
        target_module = self.graph.get_submodule(target.target)
        if isinstance(target_module, nn.BatchNorm1d):
            leftover = torch.where(mask, 1, 0).sum()
            new_layer = nn.BatchNorm1d(num_features=int(leftover))
            with torch.no_grad():
                new_layer.weight.data = target_module.weight.data[mask]
                new_layer.bias.data = target_module.bias.data[mask]
                new_layer.running_mean = target_module.running_mean.data[mask]
                new_layer.running_var = target_module.running_var.data[mask]
                if target_module.num_batches_tracked is not None:
                    new_layer.num_batches_tracked = (
                        target_module.num_batches_tracked.data
                    )
                self.replace(target, lambda: new_layer)
