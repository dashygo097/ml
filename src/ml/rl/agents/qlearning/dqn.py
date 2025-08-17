from typing import Any, Dict

import torch
from torch import nn

from ...envs import BaseDiscreteEnv
from ..base import RLAgent


class DeepQLearning(RLAgent):
    def __init__(
        self,
        env: BaseDiscreteEnv,
        dqn: nn.Module,
        init_epsilon: float = 1.0,
        final_epsilon: float = 0.0,
        epsilon_decay: float = 0.999,
        discount_rate: float = 0.99,
    ):
        super().__init__(env, discount_rate)
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon = init_epsilon

        observation_space = env.get_obs_shape()
        action_space = env.get_act_shape()

        self.dqn = dqn

    def forward(self, obs: Dict[str, Any]) -> int: ...

    def update(self, obs: Dict[str, Any], action: int, **kwargs) -> Dict[str, Any]: ...
