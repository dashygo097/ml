from typing import Dict, List

import numpy as np
import torch
from torch import nn

from ...utils import load_yaml


class ImageGANConfig:
    def __init__(self, config: str | Dict) -> None:
        if isinstance(config, str):
            self.config: Dict = load_yaml(config)
        else:
            self.config: Dict = config

        # Overall settings
        self.latent_dim: int = self.config["latent_dim"]

        # Generator
        self.generator_config: Dict = self.config["generator"]

        # Discriminator
        self.discriminator_config: Dict = self.config["discriminator"]

        self.dis_use_minibatch: bool = self.config["discriminator"].get(
            "use_minibatch", True
        )
        self.dis_minibatch_dim: int = self.config["discriminator"].get(
            "minibatch_dim", 64
        )
        self.dis_minibatch_inner_dim: int = self.config["discriminator"].get(
            "minibatch_inner_dim", 16
        )

        # IO settings
        self.n_channels = self.config["n_channels"]
        self.img_shape: List[int] = self.config["img_shape"]
        self.io_dim: int = int(np.prod(self.img_shape))


class ImageGAN(nn.Module):
    def __init__(self, config: ImageGANConfig) -> None:
        super().__init__()

        from .dcgan import DCGANConfig, DCGANDiscriminator, DCGANGenerator
        from .gan import GANConfig, GANDiscriminator, GANGenerator

        self.config = config

        if self.config.generator_config["type"] == "gan":
            self.config = GANConfig(self.config.config)
            self.generator = GANGenerator(self.config)
        elif self.config.generator_config["type"] == "dcgan":
            self.config = DCGANConfig(self.config.config)
            self.generator = DCGANGenerator(self.config)
        else:
            raise ValueError(
                f"[ERROR] Unsupported generator type: {self.config.generator_config['type']}"
            )

        if self.config.discriminator_config["type"] == "dcgan":
            self.config = DCGANConfig(self.config.config)
            self.discriminator = DCGANDiscriminator(self.config)
        elif self.config.discriminator_config["type"] == "gan":
            self.config = GANConfig(self.config.config)
            self.discriminator = GANDiscriminator(self.config)
        else:
            raise ValueError(
                f"[ERROR] Unsupported discriminator type: {self.config.discriminator_config['type']}"
            )

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def set_generator(self, generator: nn.Module) -> None:
        self.generator = generator

    def set_discriminator(self, discriminator: nn.Module) -> None:
        self.discriminator = discriminator
