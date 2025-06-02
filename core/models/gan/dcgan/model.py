import torch
from torch import nn

from .config import DCGANConfig
from .parts import DCGANDiscriminator, DCGANGenerator


class DCGAN(nn.Module):
    def __init__(self, config: DCGANConfig) -> None:
        super().__init__()
        self.config = config
        self.generator = DCGANGenerator(config)
        self.discriminator = DCGANDiscriminator(config)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def set_generator(self, generator: nn.Module) -> None:
        self.generator = generator

    def set_discriminator(self, discriminator: nn.Module) -> None:
        self.discriminator = discriminator
