from typing import Callable, List, Optional, overload

import matplotlib.pyplot as plt
from torch import nn
from torch.fx import GraphModule, symbolic_trace
from torch.fx.node import Node


class Tracer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.trace()

    def trace(self):
        self.graph: GraphModule = symbolic_trace(self.model)

    def summary(self) -> None:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        print(self.graph)

    def parse(self, folder: str = "edited", module_name: Optional[str] = None) -> None:
        if module_name is None:
            self.graph.to_folder(folder, "Traced" + self.model.__class__.__name__)
        else:
            self.graph.to_folder(folder, module_name)

    @overload
    def fetch(self, target: Optional[Node] | Optional[str]) -> Optional[Node]:
        return self.fetch(target)

    @overload
    def fetch(self, target: nn.Module) -> Optional[Node]:
        return self.fetch(target)

    @overload
    def fetch_neighbors(self, target: Optional[Node] | Optional[str]) -> List[Node]:
        return self.fetch_neighbors(target)

    @overload
    def fetch_neighbors(self, target: nn.Module) -> List[Node]:
        return self.fetch_neighbors(target)

    @overload
    def replace(
        self,
        target: Optional[Node] | Optional[str],
        new_constructor: Callable[[], nn.Module],
    ) -> List[Node]:
        return self.replace(target, new_constructor)

    @overload
    def replace(
        self, target: nn.Module, new_constructor: Callable[[], nn.Module]
    ) -> List[Node]:
        return self.replace(target, new_constructor)

    @overload
    def replace(
        self, target: type, new_constructor: Callable[[], nn.Module]
    ) -> List[Node]:
        return self.replace(target, new_constructor)

    def draw_weight_distribution(self, bins=256, count_nonzero_only=False):
        fig, axes = plt.subplots(3, 3, figsize=(10, 6))
        axes = axes.ravel()
        plot_index = 0
        for name, param in self.model.named_parameters():
            if param.dim() > 1:
                ax = axes[plot_index]
                if count_nonzero_only:
                    param_cpu = param.detach().view(-1).cpu()
                    param_cpu = param_cpu[param_cpu != 0].view(-1)
                    ax.hist(param_cpu, bins=bins, density=True, color="blue", alpha=0.5)
                else:
                    param_cpu = param.detach().view(-1).cpu()
                    ax.hist(
                        param_cpu,
                        bins=bins,
                        density=True,
                        color="blue",
                        alpha=0.5,
                    )
                ax.set_xlabel(name)
                ax.set_ylabel("density")
                plot_index += 1
        fig.suptitle("Histogram of Weights")
        fig.tight_layout()
        fig.subplots_adjust(top=0.925)
        plt.show()

    def fetch(self, target: Optional[Node] | Optional[str] | nn.Module):
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        target_node = None

        if target is None:
            raise ValueError("Target cannot be None.")

        elif isinstance(target, Node):
            for node in self.graph.graph.nodes:
                if node.op == "call_module" and node == target:
                    target_node = node
                    break

        elif isinstance(target, str):
            for node in self.graph.graph.nodes:
                if node.op == "call_module" and str(node) == target:
                    target_node = node
                    break

        elif isinstance(target, nn.Module):
            # NOTE: Return the first one if there are multiple matches
            for node in self.graph.graph.nodes:
                if node.op == "call_module":
                    submod = self.graph.get_submodule(node.target)
                    if submod is target:
                        target_node = node
                        break

        return target_node

    def fetch_neighbors(
        self, target: Optional[Node] | Optional[str] | nn.Module
    ) -> List[Node]:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        target_node = self.fetch(target)
        if target_node is None:
            return []

        connected_submodules = []

        if target is None:
            raise ValueError("Target cannot be None.")

        for node in self.graph.graph.nodes:
            if node.op == "call_module" and node != target_node:
                if any(n == target_node for n in node.all_input_nodes):
                    connected_submodules.append(node)
                elif any(n == node for n in target_node.all_input_nodes):
                    connected_submodules.append(node)

        return connected_submodules

    def replace(
        self,
        target: Optional[Node] | Optional[str] | nn.Module | type,
        new_constructor: Callable[[], nn.Module],
    ) -> List[Node]:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        replaced = []
        if target is None:
            raise ValueError("Target cannot be None.")

        elif isinstance(target, Node):
            replaced = self.replace_node(target, new_constructor)

        elif isinstance(target, str):
            target = self.fetch(target)
            replaced = self.replace_node(target, new_constructor)

        elif isinstance(target, nn.Module):
            replaced = self.replace_module(target, new_constructor)

        else:
            replaced = self.replace_typed(target, new_constructor)

        return replaced

    def replace_node(
        self, old_node: Optional[Node], new_constructor: Callable[[], nn.Module]
    ) -> List[Node]:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        replaced = []
        for node in self.graph.graph.nodes:
            if node.op == "call_module" and node == old_node:
                new_mod = new_constructor()
                self.graph.add_submodule(node.target, new_mod)
                replaced.append(node.target)
                break

        self.graph.recompile()
        return replaced

    def replace_module(
        self, old_module: nn.Module, new_constructor: Callable[[], nn.Module]
    ) -> List[Node]:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        replaced = []
        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                submod = self.graph.get_submodule(node.target)
                if submod is old_module:
                    new_mod = new_constructor()
                    self.graph.add_submodule(node.target, new_mod)
                    replaced.append(node.target)

        self.graph.recompile()
        return replaced

    def replace_typed(
        self, old_type: type, new_constructor: Callable[[], nn.Module]
    ) -> List[Node]:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        replaced = []
        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                submod = self.graph.get_submodule(node.target)
                if isinstance(submod, old_type):
                    self.graph.add_submodule(node.target, new_constructor())
                    replaced.append(node.target)

        self.graph.recompile()
        return replaced
