from typing import Any, Dict, List, Tuple

import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]) -> None:
        self.capacity: int = capacity
        self.obs_shape: Tuple[int, ...] = obs_shape
        self.buffer: List[Dict[str, Any]] = []
        self.pos: int = 0

    def append(self, transition: Dict[str, Any]) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, Any]:
        if batch_size > len(self.buffer):
            raise ValueError("Batch size exceeds the number of elements in the buffer")

        ret_dict = {}

        indices = torch.randint(0, len(self.buffer), (batch_size,))
        keys = self.buffer[0].keys()

        for key in keys:
            ret_dict[key] = torch.stack(
                [torch.tensor(self.buffer[idx][key]) for idx in indices]
            )

        return ret_dict

    def __len__(self) -> int:
        return len(self.buffer)
