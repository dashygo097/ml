from typing import Dict, List

import numpy as np

from ....utils import load_yaml


class DCGANConfig:
    def __init__(self, config_path: str) -> None:
        self.config: Dict = load_yaml(config_path)

        # Overall settings
        self.latent_dim: int = self.config["latent_dim"]

        # Generator
        self.n_gen_layers: int = self.config["generator"]["n_layers"]
        self.gen_hidden_channels: int = self.config["generator"]["hidden_channels"]
        self.gen_kernel_size: int = self.config["generator"].get("kernel_size", 3)
        self.gen_dropout: float = self.config["generator"].get("dropout", 0.0)

        # Discriminator
        self.n_dis_layers: int = self.config["discriminator"]["n_layers"]
        self.dis_hidden_channels: int = self.config["discriminator"]["hidden_channels"]
        self.dis_kernel_size: int = self.config["discriminator"].get("kernel_size", 3)
        self.dis_dropout: float = self.config["discriminator"].get("dropout", 0.0)

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
