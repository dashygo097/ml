from typing import Callable, Optional

import matplotlib.pyplot as plt
from torch import nn
from torch.fx import GraphModule, symbolic_trace


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
