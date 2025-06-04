from abc import ABC, abstractmethod

import torch
from torch import nn


class Evaluator(ABC):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def bench(self, benchmark: str): ...
