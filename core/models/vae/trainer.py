from typing import Dict, List, Tuple

from ...trainer import Trainer, TrainerArgs


class VAETrainerArgs(TrainerArgs):
    def __init__(self, path: str):
        super(VAETrainerArgs, self).__init__(path)


class VAETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(VAETrainer, self).__init__(*args, **kwargs)

    def step(self, batch) -> Dict:
        self.optimizer.zero_grad()
        if isinstance(batch, Tuple | List):
            batch = batch[0]
        input = batch.to(self.device)

        output, mean, var = self.model(input)
        output, mean, var = (
            output.to(self.device),
            mean.to(self.device),
            var.to(self.device),
        )
        loss = self.criterion(input, output, mean, var)

        loss.backward()

        self.optimizer.step()

        return {"loss": loss}

    def step_info(self, result: Dict) -> None:
        if f"epoch {self.n_epochs}" not in self.logger:
            self.logger[f"epoch {self.n_epochs}"] = {}
            self.logger[f"epoch {self.n_epochs}"]["loss"] = 0.0

        self.logger[f"epoch {self.n_epochs}"]["loss"] += float(result["loss"])

    def epoch_info(self) -> None:
        self.logger[f"epoch {self.n_epochs}"]["loss"] /= (
            len(self.data_loader) * self.args.batch_size
        )
