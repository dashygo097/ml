import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from termcolor import colored
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from .base_trainer import CNNTrainArgs, CNNTrainer


class TripletDataset(Dataset):
    def __init__(self, ds, transform=None) -> None:
        super().__init__()
        self.ds = ds
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.ds):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, label = self.ds[idx]

        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.class_to_indices[label])
        positive, _ = self.ds[positive_idx]

        negetive_label = random.choice(
            [_label for _label in self.classes if _label != label]
        )
        negative_idx = random.choice(self.class_to_indices[negetive_label])
        negative, _ = self.ds[negative_idx]

        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        return anchor, positive, negative


class SimCLRTrainArgs(CNNTrainArgs):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.contrast_margin = self.args["contrast"].get("margin", 0.5)
        self.contrast_weight = self.args["contrast"].get("weight", 1.0)


class SimCLRTrainer(CNNTrainer):
    def __init__(
        self,
        model: nn.Module,
        ds,
        criterion,
        args: SimCLRTrainArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ):
        super().__init__(model, ds, criterion, args, optimizer, scheduler, valid_ds)

    def step(
        self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]
    ) -> Dict[str, Any]:
        anchored, positive, negative = batch
        anchored = anchored.to(self.device, dtype=torch.float32)
        positive = positive.to(self.device, dtype=torch.float32)
        negative = negative.to(self.device, dtype=torch.float32)

        self.optimizer.zero_grad()

        anchor_out = self.model(anchored).view(anchored.shape[0], -1)
        positive_out = self.model(positive).view(positive.shape[0], -1)
        negative_out = self.model(negative).view(negative.shape[0], -1)

        loss = self.criterion(anchor_out, positive_out, negative_out)

        loss.backward()

        self.optimizer.step()

        return {"loss": loss.item()}

    def step_info(self, result: Dict[str, Any]) -> None:
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) + result["loss"]},
            index=self.n_epochs,
        )

    def epoch_info(self) -> None:
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) / len(self.data_loader)},
            index=self.n_epochs,
        )
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['loss']}"
        )
