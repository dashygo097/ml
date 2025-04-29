from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ...trainer import Trainer, TrainerArgs


class GANTrainerArgs(TrainerArgs):
    def __init__(self, path: str):
        super().__init__(path)
        self.beta_1: float = self.args.get("betas", 0.5)
        self.beta_2: float = self.args.get("betas", 0.999)
        self.betas = (self.beta_1, self.beta_2)

        self.noise_stddev: float = self.args.get("noise_stddev", 0.05)


class GANTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        args: GANTrainerArgs,
        optimizer=None,
        criterion=None,
    ):
        super().__init__(model, dataset, criterion, args, optimizer)
        self.set_optimizer(optimizer)
        self.set_criterion(criterion)

    def set_optimizer(self, optimizer=None):
        if optimizer is None:
            self.optimizer_G = torch.optim.Adam(
                self.model.generator.parameters(),
                lr=self.args.lr,
                betas=self.args.betas,
            )
            self.optimizer_D = torch.optim.Adam(
                self.model.discriminator.parameters(),
                lr=self.args.lr * 2,
                betas=self.args.betas,
            )
        elif isinstance(optimizer, List):
            self.optimizer_G = optimizer[0]
            self.optimizer_D = optimizer[1]

        else:
            raise ValueError("Invalid optimizer")

    def set_criterion(self, criterion=None):
        if criterion is None:
            self.criterion = [nn.BCELoss(), nn.BCELoss()]
            self.criterion_G = self.criterion[0]
            self.criterion_D = self.criterion[1]

        elif isinstance(criterion, List):
            self.criterion = criterion
            self.criterion_G = self.criterion[0]
            self.criterion_D = self.criterion[1]

        else:
            raise ValueError("Invalid criterion")

    def step(self, batch: Tuple | List) -> Dict:
        B, Total = batch[0].shape
        batch = (
            batch[0]
            .reshape(
                B,
                self.model.config.n_channels,
                self.model.config.img_shape[0],
                self.model.config.img_shape[1],
            )
            .to(self.device)
        )

        r_labels = torch.ones(B, dtype=torch.float32, device=self.device) - 0.1
        f_labels = torch.zeros(B, dtype=torch.float32, device=self.device) + 0.1

        self.optimizer_D.zero_grad()
        r_preds = self.model.discriminate(batch)
        r_loss = self.criterion_D(r_preds, r_labels)

        z = torch.randn(
            B,
            self.model.config.n_channels,
            self.model.config.latent_dim,
            device=self.device,
        )
        f_images = self.model.generate(z)
        f_preds = self.model.discriminate(f_images.detach())
        fake_loss = self.criterion_D(f_preds, f_labels)

        d_loss = r_loss + fake_loss
        d_loss.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()

        f_preds = self.model.discriminate(f_images)
        g_loss = self.criterion_G(f_preds, r_labels)
        g_loss.backward()
        self.optimizer_G.step()

        return {"g_loss": d_loss.item(), "d_loss": g_loss.item()}

    def step_info(self, result: Dict) -> None:
        if f"epoch {self.n_epochs}" not in self.logger:
            self.logger[f"epoch {self.n_epochs}"] = {}
            self.logger[f"epoch {self.n_epochs}"]["g_loss"] = 0.0
            self.logger[f"epoch {self.n_epochs}"]["d_loss"] = 0.0

        self.logger[f"epoch {self.n_epochs}"]["g_loss"] += float(result["g_loss"])
        self.logger[f"epoch {self.n_epochs}"]["d_loss"] += float(result["d_loss"])

    def epoch_info(self) -> None:
        self.logger[f"epoch {self.n_epochs}"]["g_loss"] /= (
            len(self.data_loader) * self.args.batch_size
        )
        self.logger[f"epoch {self.n_epochs}"]["d_loss"] /= (
            len(self.data_loader) * self.args.batch_size
        )
