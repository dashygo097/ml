from typing import Optional

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from ...sys import Tracer

SN_TYPES = set(
    [
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.Linear,
        nn.Embedding,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
    ]
)


class SNWrapper(Tracer):
    def __init__(self, discriminator: nn.Module):
        super().__init__(discriminator)

    def parse(
        self, folder: str = "output/sn", module_name: Optional[str] = None
    ) -> None:
        if module_name is None:
            self.graph.to_folder(folder, self.model.__class__.__name__ + "_SN")
        else:
            self.graph.to_folder(folder, module_name)

    def apply(self, n_power_iterations: int = 1):
        for node in self.graph.graph.nodes:
            if node.op == "call_module":
                submod = self.graph.get_submodule(str(node.target))
                if isinstance(submod, tuple(SN_TYPES)):
                    self.replace(
                        node,
                        lambda: spectral_norm(
                            submod, n_power_iterations=n_power_iterations
                        ),
                    )

        self.graph.recompile()
