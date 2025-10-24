from typing import Optional, overload

import torch
from torch import nn

from ..quantizer import Quantizer


class AWQ(Quantizer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def quantize(self, target: Optional[str] | type): ...
