import os
from abc import ABC, abstractmethod
from typing import Optional, overload

import torch
from termcolor import colored
from torch import nn

from ..editor import Editor


class Quantizer(ABC, Editor):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def save(self, save_dict: str, name: str = "quantized_model") -> None:
        os.makedirs(save_dict, exist_ok=True)
        path = save_dict + "/" + name + ".pt"
        torch.save(self.model.state_dict(), path)
        print(
            "[INFO] Model saved at: "
            + colored(path, "light_green", attrs=["underline"])
            + "!"
        )

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
