import os
import re
from typing import Callable, List, Optional, overload

from termcolor import colored
from torch import nn
from torch.fx import Node

from .tracer import Tracer


class Editor(Tracer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def parse(
        self, folder: str = "output/traced", module_name: Optional[str] = None
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if module_name is None:
            self.graph.to_folder(folder, "Edited" + self.model.__class__.__name__)
        else:
            self.graph.to_folder(folder, module_name)

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

    def freeze(self, target: Optional[Node] | Optional[str] | type) -> None:
        return self._freeze(target, make_frozen=True)

    def unfreeze(self, target: Optional[Node] | Optional[str] | type) -> None:
        return self._freeze(target, make_frozen=False)

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
        # FIXME: This method may not work as expected, so this feature is temporarily disabled.
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

    def _freeze(
        self,
        target: Optional[Node] | Optional[str] | nn.Module | type,
        make_frozen: bool = True,
    ) -> None:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        if target is None:
            raise ValueError(colored("Target cannot be None.", "red", attrs=["bold"]))

        elif isinstance(target, Node):
            for node in self.graph.graph.nodes:
                if node.op == "call_module" and node == target:
                    submod = self.graph.get_submodule(str(node.target))
                    for param in submod.parameters():
                        param.requires_grad = not make_frozen

        elif isinstance(target, str):
            target = self.fetch(target)
            for node in self.graph.graph.nodes:
                if node.op == "call_module" and node == target:
                    submod = self.graph.get_submodule(str(node.target))
                    for param in submod.parameters():
                        param.requires_grad = not make_frozen

        elif isinstance(target, nn.Module):
            # FIXME: This method may not work as expected, so this feature is temporarily disabled.
            for node in self.graph.graph.nodes:
                if node.op == "call_module":
                    submod = self.graph.get_submodule(str(node.target))
                    if submod is target:
                        for param in submod.parameters():
                            param.requires_grad = not make_frozen

        self.graph.recompile()
