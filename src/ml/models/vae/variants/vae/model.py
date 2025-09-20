from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class VAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class VAEDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int = 100,
    ) -> None:
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(input_dim, hidden_dim, latent_dim)

        self.mean_layer = nn.Linear(latent_dim, 2)
        self.var_layer = nn.Linear(latent_dim, 2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = self.encoder(x)
        mean = self.mean_layer(X)
        var = self.var_layer(X)
        z = mean + torch.randn_like(var) * var
        X = self.decoder(z)
        return X, mean, var

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, var = self.mean_layer(self.encoder(x)), self.var_layer(self.encoder(x))
        return mean, var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def ELBOloss(
    input: torch.Tensor, output: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    KLD = -0.5 * torch.sum(1 + var - mean**2 - torch.exp(var))
    mse_loss = F.mse_loss(input, output, reduction="sum")
    return KLD + mse_loss
