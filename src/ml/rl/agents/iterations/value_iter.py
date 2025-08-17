import random
from typing import Any, Dict, Optional

import torch
from torch import nn

from ...envs import BaseDiscreteEnv
from ..base import RLAgent


class ValueIter(RLAgent):
    def __init__(
        self,
        env: BaseDiscreteEnv,
        init_epsilon: float,
        final_epsilon: float,
        epsilon_decay: float,
        discount_rate: float = 0.99,
    ) -> None:
        super().__init__(env, discount_rate)

        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon = init_epsilon

        state_shape = env.get_obs_shape()
        act_shape = env.get_act_shape()

        self.state_values = nn.Parameter(
            torch.zeros(state_shape, dtype=torch.float32), requires_grad=False
        )
        self.policy = nn.Parameter(
            torch.zeros(state_shape + act_shape, dtype=torch.float32),
            requires_grad=False,
        )

    def reset(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(options)

    def forward(self, obs: Dict) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = obs["state"]
            action_values = self.policy[state]
            return int(torch.argmax(action_values).item())

    def policy_update(self) -> None: ...

    def value_update(self) -> None: ...

    def update_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
