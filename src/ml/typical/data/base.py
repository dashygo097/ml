from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class BaseDataset(ABC, Dataset):
    def __init__(
        self, root: str, split: str = "train", transform: Callable = T.ToTensor()
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


class BaseIterator:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Any:
        while True:
            yield self.dataset
            break
