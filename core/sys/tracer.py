from typing import Callable

from torch import nn
from torch.fx import GraphModule, symbolic_trace


class Tracer:
    def __init__(self, model: nn.Module):
        self.model: nn.Module = model
        self.trace()

    def trace(self):
        self.graph: GraphModule = symbolic_trace(self.model)

    def summary(self) -> None:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        print(self.graph)

    def replace_typed(
        self, old_type: type, new_constructor: Callable[[], nn.Module]
    ) -> int:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        replaced = 0
        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                submod = self.graph.get_submodule(node.target)
                if isinstance(submod, old_type):
                    self.graph.add_submodule(node.target, new_constructor())
                    replaced += 1

        self.graph.recompile()
        return replaced

    def replace_module(
        self, old_module: nn.Module, new_constructor: Callable[[], nn.Module]
    ) -> int:
        if self.graph is None:
            raise ValueError("Model has not been traced yet.")

        replaced = 0
        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                submod = self.graph.get_submodule(node.target)
                if submod is old_module:
                    new_mod = new_constructor()
                    self.graph.add_submodule(node.target, new_mod)
                    replaced += 1

        self.graph.recompile()
        return replaced
