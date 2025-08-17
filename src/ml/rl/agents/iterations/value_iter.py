import random
from typing import Any, Dict, Optional

import torch

from ...envs import BaseEnv
from ..base import RLAgent


class ValueIter(RLAgent):
    def __init__(
        self,
        env: BaseEnv,
        init_epsilon: float,
        final_epsilon: float,
        epsilon_decay: float,
        discount_rate: float = 0.99,
    ) -> None:
        super().__init__(env, discount_rate)

        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon = init_epsilon

        self.state_shape = env.get_obs_shape()
        self.action_shape = env.get_act_shape()

        self.register_buffer(
            "state_values", torch.zeros(self.state_shape, dtype=torch.float32)
        )

    def reset(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(options)

    def forward(self, obs: Dict) -> int: ...

    def policy_update(self) -> None: ...

    def value_update(self) -> None: ...

    def update_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
