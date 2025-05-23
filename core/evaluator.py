from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from .utils import load_yaml


class EvaluateArgs:
    def __init__(self, path: str):
        self.args = load_yaml(path)
        self.batch_size = self.args.get("batch_size", 16)
        self.is_shuffle = self.args.get("is_shuffle", False)


class Evaluator(ABC):
    def __init__(
        self,
        model: nn.Module,
        ds,
        criterion,
        device: str,
        args: EvaluateArgs,
    ):
        self.model = model
        self.criterion = criterion
        self.args = args

        self.logger = {}

        self.set_dataset(ds)
        self.set_device(device)

    def set_dataset(self, dataset) -> None:
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.is_shuffle,
        )

    def set_device(self, device) -> None:
        if device is None:
            self.device = torch.device(
                "mps"
                if torch.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        elif isinstance(device, str):
            self.device = torch.device(device)

        elif isinstance(device, torch.device):
            self.device = device

        else:
            raise ValueError("Invalid device")

    @abstractmethod
    def step(self, batch) -> Dict: ...
    def step_info(self, result: Dict) -> None: ...

    def evaluate(self) -> None:
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(
                    self.data_loader,
                    total=len(self.data_loader),
                    desc="Evaluating",
                    leave=False,
                )
            ):
                step_result = self.step(batch)
                self.step_info(step_result)
