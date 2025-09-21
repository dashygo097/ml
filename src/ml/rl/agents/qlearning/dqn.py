import random
from typing import Any, Dict, Tuple

import torch
from torch import nn

from ...envs import BaseDiscreteEnv
from ..base import RLAgent


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.buffer = []
        self.position: int = 0

    def push(self, transition: Dict[str, Any]) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Any:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


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
        self.final_epsilon: float = final_epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon: float = init_epsilon

        observation_space: Tuple[int, ...] = env.get_obs_shape()
        action_space: Tuple[int, ...] = env.get_act_shape()

        self.dqn: nn.Module = dqn

    def forward(self, obs: Dict[str, Any]) -> int: ...

    def update(self, obs: Dict[str, Any], action: int, **kwargs) -> Dict[str, Any]: ...
