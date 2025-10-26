from typing import List, Optional, Union

from torch import nn

from ..ao import AOPTQ


class AWQ(AOPTQ):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def quantize(self, target: Optional[Union[str, type]] = None) -> List[str]: ...
