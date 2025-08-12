from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class RLAgent(ABC, nn.Module):
    def __init__(self, discount_rate: float) -> None:
        super().__init__()
        self.discount_rate: float = discount_rate

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    def update(
        self,
        obs: Any,
        action: Any,
        reward: float,
        terminated: bool,
        next_obs: Any,
    ) -> Any: ...

    @abstractmethod
    def get_action(self, obs: Any) -> Any: ...
