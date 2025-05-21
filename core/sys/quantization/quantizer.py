import os
from typing import Optional, overload

from torch import nn
from torch.fx.node import Node

from ..editor import Editor


class Quantizer(Editor):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def parse(self, folder: str = "output/traced", module_name: Optional[str] = None):
        os.makedirs(folder, exist_ok=True)
        if module_name is None:
            self.graph.to_folder(folder, "Quantized" + self.model.__class__.__name__)

        else:
            self.graph.to_folder(folder, module_name)

    @overload
    def quantize(self, target: Optional[Node], dtype: str = "int8"):
        return self.quantize(target, dtype)

    @overload
    def quantize(self, target: Optional[str], dtype: str = "int8"):
        return self.quantize(target, dtype)

    @overload
    def quantize(self, target: type, dtype: str = "int8"):
        return self.quantize(target, dtype)

    def quantize(
        self, target: Optional[Node] | Optional[str] | type, dtype: str = "int8"
    ):
        if target is None:
            raise ValueError("Target cannot be None")

        elif isinstance(target, Node):
            ...

        elif isinstance(target, str):
            ...

        elif isinstance(target, type):
            ...
