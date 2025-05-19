from typing import Dict, List

import numpy as np

from ....utils import load_yaml


class GANConfig:
    def __init__(self, config_path: str) -> None:
        self.config: Dict = load_yaml(config_path)

        # Overall settings
        self.latent_dim: int = self.config["latent_dim"]

        # Generator
        self.n_gen_layers: int = self.config["generator"]["n_layers"]
        self.gen_hidden_dim: int = self.config["generator"]["hidden_dim"]
        self.gen_dropout: float = self.config["generator"]["dropout"]

        # Discriminator
        self.n_dis_layers: int = self.config["discriminator"]["n_layers"]
        self.dis_hidden_dim: int = self.config["discriminator"]["hidden_dim"]
        self.dis_dropout: float = self.config["discriminator"]["dropout"]

        # IO settings
        self.n_channels = self.config["n_channels"]
        self.img_shape: List[int] = self.config["img_shape"]
        self.io_dim: int = int(np.prod(self.img_shape))
