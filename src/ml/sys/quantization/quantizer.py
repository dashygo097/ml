from abc import ABC, abstractmethod
from typing import Optional, overload

from torch import nn

from ..editor import Editor


class Quantizer(ABC, Editor):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def save(self, path: str = "./checkpoints/quantized_model.pth") -> None:
        super().save(path)

    @overload
    def quantize(self, target: Optional[str]):
        return self.quantize(target)

    @overload
    def quantize(self, target: type):
        return self.quantize(target)

    @abstractmethod
    def quantize(self, target: Optional[str] | type):
        if target is None:
            ...

        elif isinstance(target, str):
            ...

        elif isinstance(target, type):
            ...
