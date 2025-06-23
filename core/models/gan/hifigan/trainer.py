from typing import Dict, List, Tuple

import torch.nn.functional as F

from ..trainer import GANTrainArgs, GANTrainer
from .model import HiFiGAN


class HiFiGANTrainArgs(GANTrainArgs):
    def __init__(self, path: str):
        super().__init__(path)


class HiFiGANTrainer(GANTrainer):
    def __init__(
        self,
        model: HiFiGAN,
        dataset,
        args: HiFiGANTrainArgs,
        optimizer=None,
        criterion=None,
    ):
        super().__init__(model, dataset, args, optimizer, criterion)

    def step(self, batch: Tuple | List) -> Dict:
        y = batch[0].to(self.device)
        y_mel = batch[1].to(self.device)

        y_gen = self.generator(batch)
        y_gen_mel = self.model.get_mel_spec(y_gen)

        self.optimizer_D.zero_grad()

        y_mpd_preds = self.model.mpd(y.detach())
        y_hat_mpd_preds = self.model.mpd(y_gen.detach())
        d_loss_1 = self.criterion_D(y_mpd_preds, y_hat_mpd_preds)
        y_msd_preds = self.model.msd(y.detach())
        y_hat_msd_preds = self.model.msd(y_gen.detach())
        d_loss_2 = self.criterion_D(y_msd_preds, y_hat_msd_preds)

        d_loss = d_loss_1 + d_loss_2

        d_loss.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        loss_mel = F.l1_loss(y_mel, y_gen_mel) * 45

        y_mpd_preds = self.model.mpd(y)
        y_msd_preds = self.model.msd(y_gen)
        g_loss_1 = self.criterion_G(y_hat_mpd_preds)
        y_msd_preds = self.model.msd(y)
        y_hat_msd_preds = self.model.msd(y_gen)
        g_loss_2 = self.criterion_G(y_hat_msd_preds)

        g_loss = g_loss_1 + g_loss_2 + loss_mel

        g_loss.backward()
        self.optimizer_G.step()

        return {d_loss: d_loss.item(), g_loss: g_loss.item()}

    def step_info(self, result: Dict) -> None:
        super().step_info(result)

    def epoch_info(self) -> None:
        super().epoch_info()
