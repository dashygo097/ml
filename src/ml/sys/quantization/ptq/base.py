from typing import Optional

import torch
import torchao as tq
from torch import nn

from ..quantizer import Quantizer


class PTQQuantizer(Quantizer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
