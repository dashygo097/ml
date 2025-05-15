import os
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple, overload

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.fx import Node
from torch.nn.utils import prune

from ..tracer import Tracer
from .utils import should_pass


class Pruner(Tracer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def parse(
        self, folder: str = "output/pruned", module_name: Optional[str] = None
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if module_name is None:
            self.graph.to_folder(folder, "Pruned" + self.model.__class__.__name__)
        else:
            self.graph.to_folder(folder, module_name)

    def update(self, target: Node, mask: torch.Tensor) -> List[Node]:
        queue = deque(self.fetch_neighbors(target)["output"])
        visited = []
        while queue:
            current_node = queue.popleft()
            if current_node in visited:
                continue
            visited.append(current_node)

            self._update_node(current_node, mask)

            current_module = self.graph.get_submodule(current_node.target)
            if not should_pass(current_module):
                for n in self.fetch_neighbors(current_node)["output"]:
                    if n not in visited:
                        queue.append(n)

        return visited

    @overload
    def prune(self, target: Optional[Node], amount: float = 0.2, n: int = 2) -> Dict:
        return self.prune(target, amount, n)

    @overload
    def prune(self, target: Optional[str], amount: float = 0.2, n: int = 2) -> Dict:
        return self.prune(target, amount, n)

    @overload
    def prune(self, target: type, amount: float = 0.2, n: int = 2) -> Dict:
        return self.prune(target, amount, n)

    def plot_sensitivity(
        self,
        target: Optional[Node] | Optional[str],
        dataloader,
        dim: int = 0,
        n: int = 2,
        arrange: Tuple[float, float] = (0.0, 1.0),
        steps: int = 10,
        save_folder: str = "./analysis/pruning",
    ) -> None:
        target = self.fetch(target)
        module = self.graph.get_submodule(str(target.target))

        self.model.eval()
        acc = []

        for i in range(steps):
            amount = arrange[0] + (arrange[1] - arrange[0]) * (i / steps)
            prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)

            accuracy = 0
            for features, labels in dataloader:
                output = self.model(features)
                accuracy += (output.argmax(dim=1) == labels).float().mean().item()
            accuracy /= len(dataloader)
            acc.append(accuracy)

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(
            [i / steps for i in range(steps)],
            acc,
            marker="o",
            markersize=5,
            label=f"{str(target)}",
        )
        ax.set_title("Sensitivity Analysis")
        ax.set_xlabel("Pruning Amount")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(arrange[0], arrange[1])
        ax.set_ylim(0, 1)
        ax.set_xticks(
            [i / 10 for i in range(int(arrange[0] * 10), int(arrange[1] * 10) + 1)]
        )
        ax.set_yticks([i / 10 for i in range(0, 11)])
        ax.legend()
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f"{save_folder}/sensitivity_{str(target)}.png")

    def prune(
        self,
        target: Optional[Node] | Optional[str] | type,
        amount: float = 0.2,
        n: int = 2,
    ) -> Dict:
        pruned_output = {}

        if target is None:
            raise ValueError("Target cannot be None.")
        elif isinstance(target, Node):
            pruned_output = self.prune_node(target, amount, n)
        elif isinstance(target, str):
            pruned_output = self.prune_node(target, amount, n)
        elif isinstance(target, type):
            pruned_output = self.prune_typed(target, amount, n)

        return pruned_output

    def prune_node(
        self, target: Optional[Node] | Optional[str], amount: float = 0.2, n: int = 2
    ) -> Dict:
        pruned_output = {"visited": []}
        target = self.fetch(target)

        if isinstance(target, Node):
            module = self.graph.get_submodule(str(target.target))
            if not self.fetch_neighbors(target)["output"]:
                return pruned_output

            mask = self._get_mask(module, amount, n)
            pruned_output[target] = torch.where(mask, 1, 0)

            self._shirink_prunable_node(target, mask)
            visited = self.update(target, mask)
            pruned_output["visited"].extend(visited)

            prune.remove(module, "weight")

        self.graph.recompile()
        return pruned_output

    def prune_typed(self, target: type, amount: float = 0.2, n: int = 2) -> Dict:
        pruned_output = {"visited": []}

        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                module = self.graph.get_submodule(str(node.target))
                if isinstance(module, target):
                    if not self.fetch_neighbors(node)["output"]:
                        return pruned_output

                    mask = self._get_mask(module, amount, n)
                    pruned_output[node] = torch.where(mask, 1, 0)

                    self._shirink_prunable_node(node, mask)
                    visited = self.update(node, mask)
                    pruned_output["visited"].extend(visited)
                    prune.remove(module, "weight")

        self.graph.recompile()
        return pruned_output

    def _get_mask(
        self, target: nn.Module, amount: float = 0.5, n: int = 2
    ) -> torch.Tensor:
        if isinstance(target, nn.Linear):
            prune.ln_structured(target, name="weight", amount=amount, n=n, dim=0)
            weight_importance = torch.norm(target.weight, p=n, dim=1)
            num_to_keep = int(target.weight.size(0) * (1 - amount))
        elif isinstance(target, nn.Conv2d):
            prune.ln_structured(target, name="weight", amount=amount, n=n, dim=0)
            weight_importance = torch.norm(
                target.weight.view(target.weight.size(0), -1), p=n, dim=1
            )
            num_to_keep = int(target.weight.size(0) * (1 - amount))
        else:
            raise TypeError(f"Unsupported module type: {type(target).__name__}")
        _, indices = torch.topk(weight_importance, num_to_keep)
        mask = torch.zeros_like(weight_importance, dtype=torch.bool)
        mask[indices] = True
        return mask

    def _shirink_prunable_node(self, target: Node, mask: torch.Tensor) -> None:
        current_module = self.graph.get_submodule(str(target.target))
        if isinstance(current_module, nn.Linear):
            self._shrink_linear_node(target, mask, is_prune=True)
        elif isinstance(current_module, nn.Conv2d):
            self._shrink_conv2d_node(target, mask, is_prune=True)

    def _update_node(self, target: Node, mask: torch.Tensor) -> None:
        current_module = self.graph.get_submodule(str(target.target))
        if isinstance(current_module, nn.Linear):
            self._shrink_linear_node(target, mask, is_prune=False)
        elif isinstance(current_module, nn.Conv2d):
            self._shrink_conv2d_node(target, mask, is_prune=False)
        elif isinstance(current_module, nn.BatchNorm1d):
            self._shrink_bn_node(target, mask, is_prune=False)
        elif isinstance(current_module, nn.BatchNorm2d):
            self._shrink_bn_node(target, mask, is_prune=False)

    def _shrink_linear_node(
        self, target: Node, mask: torch.Tensor, is_prune: bool
    ) -> None:
        target_module = self.graph.get_submodule(str(target.target))
        leftover = torch.where(mask, 1, 0).sum()
        if isinstance(target_module, nn.Linear):
            new_layer = nn.Linear(
                target_module.in_features
                if is_prune
                else int(leftover) * target_module.in_features // mask.shape[0],
                int(leftover) if is_prune else target_module.out_features,
                bias=target_module.bias is not None,
            )

            with torch.no_grad():
                if is_prune:
                    new_layer.weight.data = target_module.weight.data[mask, :]
                    if target_module.bias is not None:
                        new_layer.bias.data = target_module.bias.data[mask]
                else:
                    scale_factor = float(mask.shape[0]) / mask.sum()
                    mask = self._adjust_mask_dimension(
                        mask, target_module.weight.shape[1]
                    )

                    new_layer.weight.data = (
                        target_module.weight.data[:, mask] * scale_factor
                    )
                    if target_module.bias is not None:
                        new_layer.bias.data = target_module.bias.data

                self.replace(target, lambda: new_layer)

    def _shrink_conv2d_node(
        self, target: Node, mask: torch.Tensor, is_prune: bool
    ) -> None:
        target_module = self.graph.get_submodule(str(target.target))
        leftover = torch.where(mask, 1, 0).sum()
        if isinstance(target_module, nn.Conv2d):
            new_layer = nn.Conv2d(
                target_module.in_channels if is_prune else int(leftover),
                int(leftover) if is_prune else target_module.out_channels,
                kernel_size=target_module.kernel_size,
                stride=target_module.stride,
                padding=target_module.padding,
                dilation=target_module.dilation,
                groups=target_module.groups,
                bias=target_module.bias is not None,
            )

            with torch.no_grad():
                if is_prune:
                    new_layer.weight.data = target_module.weight.data[mask, :, :, :]
                    if target_module.bias is not None:
                        new_layer.bias.data = target_module.bias.data[mask]
                else:
                    scale_factor = (
                        float(target_module.weight.data.shape[1]) / mask.sum()
                    )
                    new_layer.weight.data = (
                        target_module.weight.data[:, mask, :, :] * scale_factor
                    )
                    if target_module.bias is not None:
                        new_layer.bias.data = target_module.bias.data
                self.replace(target, lambda: new_layer)

    def _shrink_bn_node(self, target: Node, mask: torch.Tensor, is_prune: bool) -> None:
        target_module = self.graph.get_submodule(str(target.target))
        leftover = torch.where(mask, 1, 0).sum()
        if isinstance(target_module, nn.BatchNorm1d):
            new_layer = nn.BatchNorm1d(num_features=int(leftover))

        elif isinstance(target_module, nn.BatchNorm2d):
            new_layer = nn.BatchNorm2d(num_features=int(leftover))

        else:
            raise TypeError(f"Unsupported module type: {type(target_module).__name__}")

        with torch.no_grad():
            new_layer.weight.data = target_module.weight.data[mask]
            new_layer.bias.data = target_module.bias.data[mask]
            new_layer.running_mean = target_module.running_mean.data[mask]
            new_layer.running_var = target_module.running_var.data[mask]
            if target_module.num_batches_tracked is not None:
                new_layer.num_batches_tracked = target_module.num_batches_tracked.data
            self.replace(target, lambda: new_layer)

    def _adjust_mask_dimension(
        self, mask: torch.Tensor, target_dim: int
    ) -> torch.Tensor:
        if mask.shape[0] == target_dim:
            return mask

        if target_dim > mask.shape[0] and target_dim % mask.shape[0] == 0:
            return mask.repeat_interleave(target_dim // mask.shape[0])

        warnings.warn(
            f"Mask dimension {mask.shape[0]} doesn't align with target dimension {target_dim}"
        )

        ratio = mask.float().mean().item()
        new_mask = torch.zeros(target_dim, dtype=torch.bool)

        keep_count = int(target_dim * ratio)
        indices = torch.randperm(target_dim)[:keep_count]
        new_mask[indices] = True

        return new_mask
