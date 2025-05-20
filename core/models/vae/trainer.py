from typing import Dict, List, Tuple

import torch
from termcolor import colored

from ...trainer import TrainArgs, Trainer


class VAETrainArgs(TrainArgs):
    def __init__(self, path: str):
        super(VAETrainArgs, self).__init__(path)


class VAETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(VAETrainer, self).__init__(*args, **kwargs)

    def step(self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]) -> Dict:
        self.optimizer.zero_grad()
        batched = batch[0].to(self.device)

        output, mean, var = self.model(batched)
        output, mean, var = (
            output.to(self.device),
            mean.to(self.device),
            var.to(self.device),
        )
        loss = self.criterion(batched, output, mean, var)

        loss.backward()

        self.optimizer.step()

        return {"loss": loss}

    def step_info(self, result: Dict) -> None:
        epoch_logger = self.logger["epoch"]
        if f"epoch {self.n_epochs}" not in epoch_logger:
            epoch_logger[f"epoch {self.n_epochs}"] = {}
            epoch_logger[f"epoch {self.n_epochs}"]["loss"] = 0.0

        epoch_logger[f"epoch {self.n_epochs}"]["loss"] += float(result["loss"].sum())

    def epoch_info(self) -> None:
        epoch_logger = self.logger["epoch"]
        epoch_logger[f"epoch {self.n_epochs}"]["loss"] /= (
            len(self.data_loader) * self.args.batch_size
        )
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {epoch_logger[f'epoch {self.n_epochs}']['loss']}"
        )
