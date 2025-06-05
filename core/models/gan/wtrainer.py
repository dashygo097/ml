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
        epoch_logger = self.logger["epoch"]
        if f"epoch {self.n_epochs}" not in epoch_logger:
            epoch_logger[f"epoch {self.n_epochs}"] = {}
            epoch_logger[f"epoch {self.n_epochs}"]["g_loss"] = 0.0
            epoch_logger[f"epoch {self.n_epochs}"]["d_loss"] = 0.0

        epoch_logger[f"epoch {self.n_epochs}"]["g_loss"] += float(result["g_loss"])
        epoch_logger[f"epoch {self.n_epochs}"]["d_loss"] += float(result["d_loss"])

    def epoch_info(self) -> None:
        epoch_logger = self.logger["epoch"]
        epoch_logger[f"epoch {self.n_epochs}"]["g_loss"] /= len(self.data_loader)
        epoch_logger[f"epoch {self.n_epochs}"]["d_loss"] /= len(self.data_loader)
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("g_loss", "yellow")
            + f": {epoch_logger[f'epoch {self.n_epochs}']['g_loss']}, "
            + colored("d_loss", "yellow")
            + f": {epoch_logger[f'epoch {self.n_epochs}']['d_loss']}"
        )
