from typing import Any


class BaseIterator:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Any:
        while True:
            yield self.dataset
            break
