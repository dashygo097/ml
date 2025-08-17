import random
from typing import Any, Dict

import torch
from torch import nn

from ...envs import BaseDiscreteEnv
from ..base import RLAgent


class QLearning(RLAgent):
    def __init__(
        self,
        env: BaseDiscreteEnv,
        init_epsilon: float = 1.0,
        final_epsilon: float = 0.0,
        epsilon_decay: float = 1.0,
        discount_rate: float = 0.99,
    ):
        super().__init__(env, discount_rate)
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon = init_epsilon

        observation_space = env.get_obs_shape()
        action_space = env.get_act_shape()

        self.q_values = nn.Parameter(
            torch.zeros(observation_space + action_space, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, obs: Dict[str, Any]) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(torch.argmax(self.q_values[tuple(obs["agent"])]))

    def update_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
