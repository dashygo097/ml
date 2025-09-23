import copy
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from termcolor import colored

from ...logger import TrainLogger
from ...trainer import TrainArgs, Trainer
from .model import ImageGAN


class ImageGANTrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)

        self.betas: Tuple[float, float] = self.optimizer_options.get(
            "betas", (0.5, 0.999)
        )
        assert "generator" in self.optimizer_options.keys(), (
            "`generator` must be specified in optimizer options!"
        )
        assert "discriminator" in self.optimizer_options.keys(), (
            "`discriminator` must be specified in optimizer options!"
        )
        self.optim_gen_options: Dict[str, Any] = self.optimizer_options.get(
            "generator", {}
        )
        self.optim_disc_options: Dict[str, Any] = self.optimizer_options.get(
            "discriminator", {}
        )

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


class ImageGANTrainer(Trainer):
    def __init__(
        self,
        model: ImageGAN,
        dataset,
        args: ImageGANTrainArgs,
        criterions: List[Callable],
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ):
        self.model: ImageGAN = model
        self.args: ImageGANTrainArgs = args
        self.generator: nn.Module = model.generator
        self.discriminator: nn.Module = model.discriminator
        self.criterions: List[Callable] = criterions

        self.set_device(args.device)
        self.set_model(model)
        self.set_dataset(dataset)
        self.set_criterions(criterions)
        self.set_optimizer(optimizer)
        self.set_scheduler(scheduler)
        self.set_valid_ds(valid_ds)

        self.n_steps: int = 0
        self.n_epochs: int = 0
        self.logger: TrainLogger = TrainLogger(self.args.log_dict)

        if self.args.enable_ema:
            self.ema_model = copy.deepcopy(self.generator)
            for param in self.ema_model.parameters():
                param.requires_grad = False

    def set_optimizer(self, optimizer):
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
        else:
            self.optimizer_G = optimizer(
                self.generator.parameters(), **self.args.optim_gen_options
            )
            self.optimizer_D = optimizer(
                self.discriminator.parameters(), **self.args.optim_disc_options
            )

    def set_scheduler(self, scheduler):
        if scheduler is None:
            scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_G, T_max=self.args.epochs
            )
            scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_D, T_max=self.args.epochs
            )
        else:
            scheduler_G = scheduler(self.optimizer_G, **self.args.scheduler_options)
            scheduler_D = scheduler(self.optimizer_D, **self.args.scheduler_options)

        self.scheduler = [scheduler_G, scheduler_D]

    def set_criterions(self, criterions) -> None:
        if criterions is None:
            self.criterions = [nn.BCELoss(), nn.BCELoss()]
            self.criterion_G = self.criterions[0]
            self.criterion_D = self.criterions[1]

        elif isinstance(criterions, List):
            self.criterions = criterions
            self.criterion_G = self.criterions[0]
            self.criterion_D = self.criterions[1]

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

    def step(
        self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]
    ) -> Dict[str, Any]:
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

        g_loss_total = torch.tensor(0.0, device=self.device)
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

            g_loss_total += g_loss

            if self.args.enable_ema:
                self.update_ema()

        g_loss_avg = g_loss_total / self.args.g2d_ratio

        return {
            "g_loss": g_loss_avg.item(),
            "d_loss": d_loss.item(),
            "r_loss": r_loss.item(),
            "f_loss": f_loss.item(),
        }

    def step_info(self, result: Dict[str, Any]) -> None:
        self.logger.op(
            "epoch",
            lambda x: {
                "g_loss": x.get("g_loss", 0) + result["g_loss"],
                "d_loss": x.get("d_loss", 0) + result["d_loss"],
                "r_loss": x.get("r_loss", 0) + result["r_loss"],
                "f_loss": x.get("f_loss", 0) + result["f_loss"],
            },
            index=self.n_epochs,
        )

    def epoch_info(self) -> None:
        self.logger.op(
            "epoch",
            lambda x: {
                "g_loss": x.get("g_loss", 0) / len(self.data_loader),
                "d_loss": x.get("d_loss", 0) / len(self.data_loader),
                "r_loss": x.get("r_loss", 0) / len(self.data_loader),
                "f_loss": x.get("f_loss", 0) / len(self.data_loader),
            },
            index=self.n_epochs,
        )

        print(
            f"(Epoch {self.n_epochs}) "
            + colored("g_loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['g_loss']:.4f}, "
            + colored("d_loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['d_loss']:.4f}, "
            + colored("r_loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['r_loss']:.4f}, "
            + colored("f_loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['f_loss']:.4f}"
        )
