from typing import Dict

from termcolor import colored
from torch import nn
from torch.utils.data import DataLoader

from ...trainer import TrainArgs, Trainer


class GNNTrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict):
        super().__init__(path_or_dict)


class GNNTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion,
        args: GNNTrainArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )

    def set_dataset(self, dataset) -> None:
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.is_shuffle,
            num_workers=self.args.num_workers,
        )

    def step(self, batch) -> Dict: ...

    def validate(self) -> None: ...

    def step_info(self, result: Dict) -> None:
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) + result["loss"].item()},
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
