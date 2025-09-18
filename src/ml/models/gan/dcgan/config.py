from typing import Dict

from ..base import ImageGANConfig


class DCGANConfig(ImageGANConfig):
    def __init__(self, config: str | Dict) -> None:
        super().__init__(config)

        # Generator
        assert "n_layers" in self.generator_config, (
            "[ERROR] 'n_layers' must be specified in the generator config."
        )
        assert "hidden_channels" in self.generator_config, (
            "[ERROR] 'hidden_channels' must be specified in the generator config."
        )
        self.n_gen_layers: int = self.generator_config["n_layers"]
        self.gen_hidden_channels: int = self.generator_config["hidden_channels"]
        self.gen_kernel_size: int = self.generator_config.get("kernel_size", 3)
        self.gen_dropout: float = self.generator_config.get("dropout", 0.0)

        # Discriminator
        assert "n_layers" in self.discriminator_config, (
            "[ERROR] 'n_layers' must be specified in the discriminator config."
        )
        assert "hidden_channels" in self.discriminator_config, (
            "[ERROR] 'hidden_channels' must be specified in the discriminator config."
        )
        self.n_dis_layers: int = self.discriminator_config["n_layers"]
        self.dis_hidden_channels: int = self.discriminator_config["hidden_channels"]
        self.dis_kernel_size: int = self.discriminator_config.get("kernel_size", 3)
        self.dis_dropout: float = self.discriminator_config.get("dropout", 0.0)

        self.dis_minibatch_out_features: int = self.discriminator_config.get(
            "minibatch_out_features", 64
        )
