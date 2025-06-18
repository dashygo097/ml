from typing import Dict, List, Tuple

import torch
from termcolor import colored

from .frontend import ImageGAN
from .trainer import GANTrainArgs, GANTrainer


class WGANTrainer(GANTrainer):
    def __init__(
        self,
        model: ImageGAN,
        dataset,
        args: GANTrainArgs,
        criterion=None,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            args=args,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            valid_ds=valid_ds,
        )

    def step(self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]) -> Dict: ...

    def step_info(self, result: Dict) -> None:
        self.logger.op(
            "epoch",
            lambda x: {
                "g_loss": x.get("g_loss", 0) + float(result["g_loss"].item()),
                "d_loss": x.get("d_loss", 0) + float(result["d_loss"].item()),
            },
        )

    def epoch_info(self) -> None:
        self.logger.op(
            "epoch",
            lambda x: {
                "g_loss": x.get("g_loss", 0) / len(self.data_loader),
                "d_loss": x.get("d_loss", 0) / len(self.data_loader),
            },
            index=self.n_epochs,
        )
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("g_loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['g_loss']}, "
            + colored("d_loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['d_loss']}"
        )
