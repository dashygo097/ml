from torch import nn

from ..quantizer import Quantizer


class PTQ(Quantizer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
