import torch
import torch.nn.functional as F


def ELBOloss(
    input: torch.Tensor, output: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    KLD = -0.5 * torch.sum(1 + var - mean**2 - torch.exp(var))
    mse_loss = F.mse_loss(input, output, reduction="sum")
    return KLD + mse_loss
