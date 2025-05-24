import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from termcolor import colored
from torch import nn
from torch.utils.data import Dataset

from .trainer import CNNTrainArgs, CNNTrainer


class TripletDataset(Dataset):
    def __init__(self, ds, transforms=None) -> None:
        super().__init__()
        self.ds = ds
        self.transforms = transforms

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
        positive = self.ds[positive_idx]

        negetive_label = random.choice(
            [_label for _label in self.classes if _label != label]
        )
        negative_idx = random.choice(self.class_to_indices[negetive_label])
        negative = self.ds[negative_idx]

        if self.transforms is not None:
            anchor = self.transforms(anchor)
            positive = self.transforms(positive)
            negative = self.transforms(negative)

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

    def step(self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]) -> Dict:
        anchored, positive, negative = batch

        self.optimizer.zero_grad()

        anchor_out = self.model(anchored).view(anchored.shape[0], -1)
        positive_out = self.model(positive).view(positive.shape[0], -1)
        negative_out = self.model(negative).view(negative.shape[0], -1)

        anchor_positive = torch.cosine_similarity(anchor_out, positive_out)
        anchor_negative = torch.cosine_similarity(anchor_out, negative_out)

        loss = self.criterion(
            anchor_positive,
            anchor_negative,
            self.args.contrast_margin,
            self.args.contrast_weight,
        )

        loss.backward()

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "anchor_positive": anchor_positive.item(),
            "anchor_negative": anchor_negative.item(),
        }

    def step_info(self, result: Dict) -> None:
        epoch_logger = self.logger["epoch"]
        if f"epoch {self.n_epochs}" not in epoch_logger:
            epoch_logger[f"epoch {self.n_epochs}"] = {}
            epoch_logger[f"epoch {self.n_epochs}"]["loss"] = 0.0
            epoch_logger[f"epoch {self.n_epochs}"]["anchor_positive"] = 0.0
            epoch_logger[f"epoch {self.n_epochs}"]["anchor_negative"] = 0.0

        epoch_logger[f"epoch {self.n_epochs}"]["loss"] += float(result["loss"].sum())
        epoch_logger[f"epoch {self.n_epochs}"]["anchor_positive"] += float(
            result["anchor_positive"].sum()
        )
        epoch_logger[f"epoch {self.n_epochs}"]["anchor_negative"] += float(
            result["anchor_negative"].sum()
        )
        self.logger["epoch"] = epoch_logger

    def epoch_info(self) -> None:
        self.logger["epoch"][f"epoch {self.n_epochs}"]["loss"] /= len(self.data_loader)
        self.logger["epoch"][f"epoch {self.n_epochs}"]["anchor_positive"] /= len(
            self.data_loader
        )
        self.logger["epoch"][f"epoch {self.n_epochs}"]["anchor_negative"] /= len(
            self.data_loader
        )
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {self.logger['epoch'][f'epoch {self.n_epochs}']['loss']}\n"
            + colored("anchor_positive", "yellow")
            + f": {self.logger['epoch'][f'epoch {self.n_epochs}']['anchor_positive']} "
            + colored("anchor_negative", "yellow")
        )
