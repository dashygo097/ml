import random
from typing import Any, Dict

import torch

from ...envs import BaseEnv
from ..base import RLAgent


class QLearning(RLAgent):
    def __init__(
        self,
        env: BaseEnv,
        init_epsilon: float,
        final_epsilon: float,
        epsilon_decay: float,
        discount_rate: float = 0.99,
    ):
        super().__init__(env, discount_rate)
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon = init_epsilon

        q_values_shape = env.get_obs_shape() + env.get_act_shape()

        self.register_buffer(
            "q_values", torch.zeros(q_values_shape, dtype=torch.float32)
        )

    def forward(self, obs: Dict) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(torch.argmax(self.q_values[tuple(obs["agent"])]))

    def update_epsilon(self) -> Any:
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
