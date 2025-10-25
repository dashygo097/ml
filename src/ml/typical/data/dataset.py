from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    def __init__(
        self, root: str, split: str = "train", transform: Optional[Callable] = None
    ):
        self.root = root
        self.split = split
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]: ...
