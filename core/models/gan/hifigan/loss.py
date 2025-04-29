from typing import List

import torch


def hifigan_d_loss(
    disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
):
    loss = torch.Tensor(0.0)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss

    return loss


def hifigan_g_loss(disc_outputs: List[torch.Tensor]):
    loss = torch.Tensor(0.0)
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        loss += l

    return loss
