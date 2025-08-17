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
        epsilon_decay: float = 0.999,
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
            agent_obs = tuple(obs["agent"])
            return int(self.q_values[agent_obs].argmax())

    def update(self, obs: Dict[str, Any], action: int, **kwargs) -> Dict[str, Any]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        q_values_index = tuple(obs["agent"]) + (action,)
        future_q_values_index = tuple(next_obs["agent"])

        with torch.no_grad():
            future_q = 0 if terminated else self.q_values[future_q_values_index].max()
            target = reward + self.discount_rate * future_q
            error = target - self.q_values[q_values_index]
            loss = error * error

            self.q_values[q_values_index] += kwargs.get("lr", 0.0) * error

        return {
            "next_obs": next_obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "loss": float(loss),
        }

    def update_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
