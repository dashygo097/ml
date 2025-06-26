from typing import Dict, List, Tuple

import torch
from torch import nn

from .base_trainer import GANTrainArgs, GANTrainer


class GANTrainArgsCT(GANTrainArgs):
    """
    GANTrainerCT is a specialized trainer for Conditional GANs (cGANs).
    It inherits from GANTrainer and is designed to handle the training of cGANs.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)


class GANTrainerCT(GANTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        args: GANTrainArgsCT,
        criterion=None,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(
            model=model,
            dataset=dataset,
            args=args,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            valid_ds=valid_ds,
        )

    def step(self, batch: Tuple[torch.Tensor, ...] | List[torch.Tensor]) -> Dict:
        batched, labels = batch
        B = batched.shape[0]

        batched = batched.reshape(
            B,
            self.model.config.n_channels,
            self.model.config.img_shape[0],
            self.model.config.img_shape[1],
        ).to(self.device)
        labels = labels.reshape(B, 1).to(self.device)

        batched += torch.randn_like(batched) * self.args.instance_noise_stddev

        r_labels = (
            torch.ones(B, dtype=torch.float32, device=self.device)
            - self.args.label_smoothing
        )
        flip_mask = torch.rand(B, device=self.device) < self.args.flip_chance
        r_labels[flip_mask] = 1 - r_labels[flip_mask]

        f_labels = (
            torch.zeros(B, dtype=torch.float32, device=self.device)
            + self.args.label_smoothing
        )
        flip_mask = torch.rand(B, device=self.device) < self.args.flip_chance
        f_labels[flip_mask] = 1 - f_labels[flip_mask]

        self.optimizer_D.zero_grad()
        r_preds = self.discriminator(batched, labels)
        r_loss = self.criterion_D(r_preds, r_labels)

        z = torch.randn(B, self.model.config.latent_dim, device=self.device)
        z += torch.randn_like(z) * self.args.latent_noise_stddev

        f_images = self.generator(z, labels)
        f_preds = self.discriminator(f_images, labels)
        f_loss = self.criterion_D(f_preds, f_labels)

        d_loss = r_loss + f_loss
        d_loss.backward()
        self.optimizer_D.step()

        g_loss_total = 0.0
        for _ in range(self.args.g2d_ratio):
            self.optimizer_G.zero_grad()

            if self.args.fresh_samples_per_g_step:
                z = torch.randn(B, self.model.config.latent_dim, device=self.device)
                z += torch.randn_like(z) * self.args.latent_noise_stddev
                f_images = self.generator(z, labels)

            f_preds = self.discriminator(f_images, labels)
            g_loss = self.criterion_G(f_preds, r_labels)
            g_loss.backward()
            self.optimizer_G.step()
            g_loss_total += g_loss.item()

            if self.args.enable_ema:
                self.update_ema()

        g_loss_avg = g_loss_total / self.args.g2d_ratio

        return {
            "g_loss": g_loss_avg,
            "d_loss": d_loss,
        }
