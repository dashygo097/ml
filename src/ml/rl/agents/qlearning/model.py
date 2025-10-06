import random
from typing import Any, Dict, Tuple

import torch
from torch import nn

from ...envs import BaseDiscreteEnv
from ..base import EpsilonGreedyAgent


class QLearning(EpsilonGreedyAgent):
    def __init__(
        self,
        env: BaseDiscreteEnv,
        init_epsilon: float = 1.0,
        final_epsilon: float = 0.0,
        epsilon_decay: float = 0.999,
        discount_rate: float = 0.99,
    ):
        super().__init__(env, init_epsilon, final_epsilon, epsilon_decay, discount_rate)

        observation_space: Tuple[int, ...] = env.get_obs_shape()
        action_space: Tuple[int, ...] = env.get_act_shape()

        self.q_values: torch.Tensor
        self.register_buffer(
            "q_values",
            torch.zeros(observation_space + action_space, dtype=torch.float32),
        )

    def forward(self, obs: Dict[str, Any]) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                agent_obs = tuple(obs["agent"])
                return int(self.q_values[agent_obs].argmax())


class DeepQLearning(EpsilonGreedyAgent):
    def __init__(
        self,
        env: BaseDiscreteEnv,
        dqn: nn.Module,
        init_epsilon: float = 1.0,
        final_epsilon: float = 0.0,
        epsilon_decay: float = 0.999,
        discount_rate: float = 0.99,
    ):
        super().__init__(env, init_epsilon, final_epsilon, epsilon_decay, discount_rate)
        self.dqn = dqn
        self.final_epsilon: float = final_epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon: float = init_epsilon

    def forward(self, obs: Dict[str, Any]) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = obs["agent"]
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                if state.ndim == 1:
                    state = state.unsqueeze(0)

                q_values = self.dqn(state)
                return int(q_values.argmax(dim=1)[0])
