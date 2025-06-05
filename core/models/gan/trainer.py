import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from termcolor import colored

from ...trainer import TrainArgs, Trainer
from .frontend import ImageGAN


class GANTrainArgs(TrainArgs):
    def __init__(self, path: str):
        super().__init__(path)
        self.beta_1: float = self.args.get("betas", 0.5)
        self.beta_2: float = self.args.get("betas", 0.999)
        self.betas = (self.beta_1, self.beta_2)

        self.enable_ema: bool = self.args["ema"].get("enable", False)
        self.ema_decay: float = self.args["ema"].get("decay", 0.999)

        self.instance_noise_stddev: float = self.args.get("instance_noise_stddev", 0)
        self.latent_noise_stddev: float = self.args.get("latent_noise_stddev", 0)
        self.label_smoothing: float = self.args.get("label_smoothing", 0.1)
        self.flip_chance: float = self.args.get("flip_chance", 0)

        self.g2d_ratio: int = self.args.get("g2d_ratio", 1)
        self.fresh_samples_per_g_step: bool = self.args.get(
            "fresh_samples_per_g_step", False
        )


class GANTrainer(Trainer):
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
        self.model = model
        self.criterion = criterion
        self.args = args
        self.generator: nn.Module = model.generator
        self.discriminator: nn.Module = model.discriminator

        self.set_device(args.device)
        self.set_model(model)
        self.set_dataset(dataset)
        self.set_criterion(criterion)
        self.set_optimizer(optimizer)
        self.set_schedulers(scheduler)
        self.set_valid_ds(valid_ds)

        self.n_steps: int = 0
        self.n_epochs: int = 0
        self.logger: Dict = {"epoch": {}, "step": {}, "valid": {}}

        if self.args.enable_ema:
            self.ema_model = copy.deepcopy(self.generator)
            for param in self.ema_model.parameters():
                param.requires_grad = False

    def set_optimizer(self, optimizer=None):
        if optimizer is None:
            self.optimizer_G = torch.optim.Adam(
                self.generator.parameters(),
                lr=self.args.lr,
                betas=self.args.betas,
            )
            self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.args.lr * 2,
                betas=self.args.betas,
            )
        elif isinstance(optimizer, List):
            self.optimizer_G = optimizer[0]
            self.optimizer_D = optimizer[1]

        else:
            raise ValueError("Invalid optimizer")

    def set_schedulers(self, schedulers: Optional[Union[List, Tuple]]):
        if schedulers is None:
            scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_G, T_max=self.args.n_epochs
            )
            scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_D, T_max=self.args.n_epochs
            )
            self.schedulers = [scheduler_G, scheduler_D]

        elif isinstance(schedulers, List):
            self.schedulers = schedulers

        elif isinstance(schedulers, Tuple):
            self.schedulers = list(schedulers)

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

    def save(self) -> None:
        os.makedirs(self.args.save_dict, exist_ok=True)
        path = self.args.save_dict + "/checkpoint_" + str(self.n_steps) + ".pt"
        torch.save(self.model.state_dict(), path)
        print(
            "[INFO] Model saved at: "
            + colored(path, "light_green", attrs=["underline"])
            + "!"
        )

        if self.args.enable_ema:
            ema_path = (
                self.args.save_dict + "/ema_checkpoint_" + str(self.n_steps) + ".pt"
            )
            torch.save(self.ema_model.state_dict(), ema_path)
            print(
                "[INFO] EMA Model saved at: "
                + colored(ema_path, "light_green", attrs=["underline"])
                + "!"
            )

    def validate(self) -> None:
        os.makedirs(
            "train_logs/validation/valid_"
            + str(self.n_epochs)
            + "x"
            + str(self.n_steps),
            exist_ok=True,
        )
        inputs = torch.randn(10, self.model.config.latent_dim).to(self.device)
        outputs = self.generator(inputs).squeeze(1)
        for i in range(10):
            output = outputs[i].cpu().detach()
            fig, ax = plt.subplots(ncols=1, figsize=(6, 5))
            ax.imshow(output.detach().numpy())
            ax.set_title("Generated Image")
            plt.savefig(
                f"train_logs/validation/valid_{self.n_epochs}x{self.n_steps}/generated_image_{i}.png"
            )
        plt.close("all")

    def update_ema(self) -> None:
        if self.args.enable_ema:
            for ema_param, model_param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data = (
                    self.args.ema_decay * ema_param.data
                    + (1 - self.args.ema_decay) * model_param.data
                )

    def step(self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]) -> Dict:
        B = batch[0].shape[0]
        batched = (
            batch[0]
            .reshape(
                B,
                self.model.config.n_channels,
                self.model.config.img_shape[0],
                self.model.config.img_shape[1],
            )
            .to(self.device)
        )
        if self.args.instance_noise_stddev > 0:
            batched += (
                torch.randn_like(batched, device=self.device)
                * self.args.instance_noise_stddev
            )

        r_labels = (
            torch.ones(B, 1, dtype=torch.float32, device=self.device)
            - self.args.label_smoothing
        )
        if self.args.flip_chance > 0:
            flip_mask = torch.rand(B, 1, device=self.device) < self.args.flip_chance
            r_labels[flip_mask] = 1 - r_labels[flip_mask]

        f_labels = (
            torch.zeros(B, 1, dtype=torch.float32, device=self.device)
            + self.args.label_smoothing
        )
        if self.args.flip_chance > 0:
            flip_mask = torch.rand(B, 1, device=self.device) < self.args.flip_chance
            f_labels[flip_mask] = 1 - f_labels[flip_mask]

        self.optimizer_D.zero_grad()
        r_preds = self.discriminator(batched)
        r_loss = self.criterion_D(r_preds, r_labels)

        z = torch.randn(
            B,
            self.model.config.latent_dim,
            device=self.device,
        )
        if self.args.latent_noise_stddev > 0:
            z += torch.randn_like(z, device=self.device) * self.args.latent_noise_stddev

        f_images = self.generator(z)
        f_preds = self.discriminator(f_images.detach())
        f_loss = self.criterion_D(f_preds, f_labels)

        d_loss = r_loss + f_loss
        d_loss.backward()
        self.optimizer_D.step()

        g_loss_total = 0
        for _ in range(self.args.g2d_ratio):
            self.optimizer_G.zero_grad()

            if self.args.fresh_samples_per_g_step:
                z = torch.randn(B, self.model.config.latent_dim, device=self.device)
                if self.args.latent_noise_stddev > 0:
                    z += (
                        torch.randn_like(z, device=self.device)
                        * self.args.latent_noise_stddev
                    )
                f_images = self.generator(z)

            f_preds = self.discriminator(f_images)
            g_loss = self.criterion_G(f_preds, r_labels)
            g_loss.backward()
            self.optimizer_G.step()

            g_loss_total += g_loss.item()

            if self.args.enable_ema:
                self.update_ema()

        g_loss_avg = g_loss_total / self.args.g2d_ratio

        return {
            "g_loss": g_loss_avg,
            "d_loss": d_loss.item(),
            "r_loss": r_loss.item(),
            "f_loss": f_loss.item(),
        }

    def step_info(self, result: Dict) -> None:
        epoch_logger = self.logger["epoch"]
        if f"epoch {self.n_epochs}" not in epoch_logger:
            epoch_logger[f"epoch {self.n_epochs}"] = {}
            epoch_logger[f"epoch {self.n_epochs}"]["g_loss"] = 0.0
            epoch_logger[f"epoch {self.n_epochs}"]["d_loss"] = 0.0
            epoch_logger[f"epoch {self.n_epochs}"]["r_loss"] = 0.0
            epoch_logger[f"epoch {self.n_epochs}"]["f_loss"] = 0.0

        epoch_logger[f"epoch {self.n_epochs}"]["g_loss"] += float(result["g_loss"])
        epoch_logger[f"epoch {self.n_epochs}"]["d_loss"] += float(result["d_loss"])
        epoch_logger[f"epoch {self.n_epochs}"]["r_loss"] += float(result["r_loss"])
        epoch_logger[f"epoch {self.n_epochs}"]["f_loss"] += float(result["f_loss"])

    def epoch_info(self) -> None:
        epoch_logger = self.logger["epoch"]
        epoch_logger[f"epoch {self.n_epochs}"]["g_loss"] /= len(self.data_loader)
        epoch_logger[f"epoch {self.n_epochs}"]["d_loss"] /= len(self.data_loader)
        epoch_logger[f"epoch {self.n_epochs}"]["r_loss"] /= len(self.data_loader)
        epoch_logger[f"epoch {self.n_epochs}"]["f_loss"] /= len(self.data_loader)
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("g_loss", "yellow")
            + f": {epoch_logger[f'epoch {self.n_epochs}']['g_loss']}, "
            + colored("d_loss", "yellow")
            + f": {epoch_logger[f'epoch {self.n_epochs}']['d_loss']}, "
            + colored("r_loss", "yellow")
            + f": {epoch_logger[f'epoch {self.n_epochs}']['r_loss']}, "
            + colored("f_loss", "yellow")
            + f": {epoch_logger[f'epoch {self.n_epochs}']['f_loss']}"
        )
