from typing import Any, Dict, List, Tuple

import torch


class ReplayBuffer:
    def __init__(
        self, capacity: int, observation_shape: Tuple[int, ...], device
    ) -> None:
        self.capacity: int = capacity
        self.observation_shape: Tuple[int, ...] = observation_shape
        self.buffer: List[Dict[str, Any]] = []
        self.pos: int = 0

        self.device = device

    def append(self, transition: Dict[str, Any]) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, Any]: ...

    def __len__(self) -> int:
        return len(self.buffer)
