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


class EpsilonGreedyAgent(RLAgent):
    def __init__(
        self,
        env: BaseEnv,
        init_epsilon: float,
        final_epsilon: float,
        epsilon_decay: float,
        discount_rate: float,
    ):
        super().__init__(env, discount_rate)
        self.final_epsilon: float = final_epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon: float = init_epsilon

    def update_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


class PolicyFn(Protocol):
    def __call__(self, obs: Any) -> Any: ...
