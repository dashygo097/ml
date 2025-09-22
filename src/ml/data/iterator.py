from abc import ABC, abstractmethod
from typing import Any


class BaseIterator(ABC):
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    @abstractmethod
    def __len__(self) -> int:
        return 1

    @abstractmethod
    def __iter__(self) -> Any:
        while True:
            yield self.dataset
            break
