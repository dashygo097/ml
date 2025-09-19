from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn

from ...utils import load_yaml
from .dcgan import (DCGANDiscriminator, DCGANDiscriminatorConfig,
                    DCGANGenerator, DCGANGeneratorConfig)
from .gan import (GANDiscriminator, GANDiscriminatorConfig, GANGenerator,
                  GANGeneratorConfig)


class ImageGANConfig:
    def __init__(self, config: str | Dict) -> None:
        self.config: Dict[str, Any] = (
            config if isinstance(config, Dict) else load_yaml(config)
        )

        # IO settings
        assert "n_channels" in self.config, (
            "[ERROR] 'n_channels' must be specified in the config."
        )
        assert "res" in self.config, "[ERROR] 'res' must be specified in the config."
        self.res: Tuple[int, int] = self.config["res"]
        self.n_channels: int = self.config["n_channels"]

        self.io_dim: int = int(np.prod(self.res))

        # Generator
        assert "generator" in self.config, (
            "[ERROR] 'generator' must be specified in the config."
        )
        assert "type" in self.config["generator"], (
            "[ERROR] 'type' must be specified in the 'generator' config."
        )
        self.generator_config: Dict[str, Any] = self.config["generator"]
        self.generator_config["res"] = self.res
        self.generator_config["n_channels"] = self.n_channels

        # Discriminator
        assert "discriminator" in self.config, (
            "[ERROR] 'discriminator' must be specified in the config."
        )
        assert "type" in self.config["discriminator"], (
            "[ERROR] 'type' must be specified in the 'discriminator' config."
        )
        self.discriminator_config: Dict[str, Any] = self.config["discriminator"]
        self.discriminator_config["res"] = self.res
        self.discriminator_config["n_channels"] = self.n_channels


class ImageGAN(nn.Module):
    def __init__(self, config: ImageGANConfig) -> None:
        super().__init__()
        self.config: Dict[str, Any] = config.config
        self.generator_config: Dict[str, Any] = config.generator_config
        self.discriminator_config: Dict[str, Any] = config.discriminator_config

        if self.generator_config["type"] == "gan":
            gen_config = GANGeneratorConfig(self.generator_config)
            self.generator = GANGenerator(gen_config)
        elif self.generator_config["type"] == "dcgan":
            gen_config = DCGANGeneratorConfig(self.generator_config)
            self.generator = DCGANGenerator(gen_config)
        else:
            raise ValueError(
                f"[ERROR] Unsupported generator type: {self.generator_config['type']}"
            )

        if self.discriminator_config["type"] == "gan":
            dis_config = GANDiscriminatorConfig(self.discriminator_config)
            self.discriminator = GANDiscriminator(dis_config)
        elif self.discriminator_config["type"] == "dcgan":
            dis_config = DCGANDiscriminatorConfig(self.discriminator_config)
            self.discriminator = DCGANDiscriminator(dis_config)
        else:
            raise ValueError(
                f"[ERROR] Unsupported discriminator type: {self.discriminator_config['type']}"
            )

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def set_generator(self, generator: nn.Module) -> None:
        self.generator = generator

    def set_discriminator(self, discriminator: nn.Module) -> None:
        self.discriminator = discriminator
