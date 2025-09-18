from typing import Dict

from ..base import ImageGANConfig


class GANConfig(ImageGANConfig):
    def __init__(self, config: str | Dict) -> None:
        super().__init__(config)

        # Generator
        assert "n_layers" in self.generator_config, (
            "[ERROR] 'n_layers' must be specified in generator_config"
        )
        assert "hidden_dim" in self.generator_config, (
            "[ERROR] 'hidden_dim' must be specified in generator_config"
        )
        self.n_gen_layers: int = self.generator_config["n_layers"]
        self.gen_hidden_dim: int = self.generator_config["hidden_dim"]
        self.gen_dropout: float = self.generator_config.get("dropout", 0.0)

        # Discriminator
        assert "n_layers" in self.discriminator_config, (
            "[ERROR] 'n_layers' must be specified in discriminator_config"
        )
        assert "hidden_dim" in self.discriminator_config, (
            "[ERROR] 'hidden_dim' must be specified in discriminator_config"
        )
        self.n_dis_layers: int = self.discriminator_config["n_layers"]
        self.dis_hidden_dim: int = self.discriminator_config["hidden_dim"]
        self.dis_dropout: float = self.discriminator_config.get("dropout", 0.0)
