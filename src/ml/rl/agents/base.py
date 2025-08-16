from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

from torch import nn

from ..envs import BaseEnv


class RLAgent(ABC, nn.Module):
    def __init__(self, env: BaseEnv, discount_rate: float = 0.99) -> None:
        super().__init__()
        self.env: BaseEnv = env
        self.discount_rate: float = discount_rate

    def reset(
        self,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.discount_rate = (
            options.get("discount_rate", self.discount_rate)
            if options
            else self.discount_rate
        )

    @abstractmethod
    def forward(self, obs: Any) -> Any: ...


class PolicyFn(Protocol):
    def __call__(self, obs: Any) -> Any: ...
