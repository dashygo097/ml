import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, overload

import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
from termcolor import colored
from torch import nn
from torch.fx import GraphModule, symbolic_trace
from torch.fx.node import Node


class Tracer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.trace()

    def trace(self):
        self.graph: GraphModule = symbolic_trace(self.model)

    def numal(self, info: bool = False) -> int:
        num_params = 0
        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                submod = self.graph.get_submodule(str(node.target))
                num_params += sum(p.numel() for p in submod.parameters())

        if info:
            print(
                colored(f"Number of parameters: {num_params}", "blue", attrs=["bold"])
            )
        return num_params

    def get_details(self, input_shape: Tuple, info: bool = False) -> Tuple[str, str]:
        warnings.warn(
            colored(
                "[WARN] Method `get_details` only returns the parmas of THE ORIGINAL MODEL`",
                "yellow",
                attrs=["bold"],
            )
        )
        macs, params = get_model_complexity_info(
            self.model,
            input_shape,
            as_strings=True,
            print_per_layer_stat=info,
            verbose=info,
        )

        if info:
            print(colored(f"MACs: {macs}", "blue", attrs=["bold"]))
            print(colored(f"Parameters: {params}", "blue", attrs=["bold"]))

        return str(macs), str(params)

    def summary(self) -> None:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        print(self.graph)
        self.numal(info=True)

    def parse(
        self, folder: str = "output/traced", module_name: Optional[str] = None
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if module_name is None:
            self.graph.to_folder(folder, "Traced" + self.model.__class__.__name__)
        else:
            self.graph.to_folder(folder, module_name)

    @overload
    def fetch(self, target: Optional[Node]) -> Optional[Node]:
        return self.fetch(target)

    @overload
    def fetch(self, target: Optional[str]) -> Optional[Node]:
        return self.fetch(target)

    @overload
    def fetch_neighbors(
        self, target: Optional[Node], follow_operators: bool = True
    ) -> Dict:
        return self.fetch_neighbors(target, follow_operators)

    @overload
    def fetch_neighbors(
        self, target: Optional[str], follow_operators: bool = True
    ) -> Dict:
        return self.fetch_neighbors(target, follow_operators)

    @overload
    def replace(
        self,
        target: Optional[Node] | Optional[str],
        new_constructor: Callable[[], nn.Module],
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

    def fetch(self, target: Optional[Node] | Optional[str]):
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        target_node = None

        if target is None:
            raise ValueError(
                colored("[ERROR] Target cannot be None.", "red", attrs=["bold"])
            )

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

        return target_node

    def fetch_neighbors(
        self,
        target: Optional[Node] | Optional[str],
        follow_operators: bool = True,
    ) -> Dict:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        target_node = self.fetch(target)
        if target_node is None:
            return {}

        connected_submodules = {"input": [], "output": []}

        if target is None:
            raise ValueError(
                colored("[ERROR] Target cannot be None.", "red", attrs=["bold"])
            )

        def find_upstream_modules(node, depth=0):
            for other_node in node.all_input_nodes:
                if other_node.op == "call_module":
                    if other_node not in connected_submodules["input"]:
                        connected_submodules["input"].append(other_node)
                elif follow_operators and other_node.op in [
                    "call_function",
                    "call_method",
                ]:
                    find_upstream_modules(other_node, depth + 1)

        def find_downstream_modules(node, depth=0):
            for other_node in self.graph.graph.nodes:
                if node in other_node.all_input_nodes:
                    if other_node.op == "call_module":
                        if other_node not in connected_submodules["output"]:
                            connected_submodules["output"].append(other_node)
                    elif follow_operators and other_node.op in [
                        "call_function",
                        "call_method",
                    ]:
                        find_downstream_modules(other_node, depth + 1)

        find_upstream_modules(target_node)
        find_downstream_modules(target_node)

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
            raise ValueError(colored("Target cannot be None.", "red", attrs=["bold"]))

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
                self.graph.add_submodule(str(node.target), new_mod)
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
                submod = self.graph.get_submodule(str(node.target))
                if submod is old_module:
                    new_mod = new_constructor()
                    self.graph.add_submodule(str(node.target), new_mod)
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
                submod = self.graph.get_submodule(str(node.target))
                if isinstance(submod, old_type):
                    self.graph.add_submodule(str(node.target), new_constructor())
                    replaced.append(node.target)

        self.graph.recompile()
        return replaced
