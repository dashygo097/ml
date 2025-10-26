from typing import Optional

from torch import nn

from ..base import PTQ


class AWQ(PTQ):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def quantize(self, target: Optional[str] | type): ...
