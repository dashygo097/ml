from typing import Dict

from ..frontend import ImageGANConfig


class DCGANConfig(ImageGANConfig):
    def __init__(self, config: str | Dict) -> None:
        super().__init__(config)

        # Generator
        self.n_gen_layers: int = self.generator_config["n_layers"]
        self.gen_hidden_channels: int = self.generator_config["hidden_channels"]
        self.gen_kernel_size: int = self.generator_config.get("kernel_size", 3)
        self.gen_dropout: float = self.generator_config.get("dropout", 0.0)

        # Discriminator
        self.n_dis_layers: int = self.discriminator_config["n_layers"]
        self.dis_hidden_channels: int = self.discriminator_config["hidden_channels"]
        self.dis_kernel_size: int = self.discriminator_config.get("kernel_size", 3)
        self.dis_dropout: float = self.discriminator_config.get("dropout", 0.0)
