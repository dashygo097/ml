from typing import Any, Dict, Tuple

import numpy as np

from .....utils import load_yaml


class GANGeneratorConfig:
    def __init__(self, config: str | Dict) -> None:
        self.config: Dict[str, Any] = (
            config if isinstance(config, Dict) else load_yaml(config)
        )
        assert "n_layers" in self.config, (
            "[ERROR] 'n_layers' must be specified in the config."
        )
        assert "latent_dim" in self.config, (
            "[ERROR] 'latent_dim' must be specified in the config."
        )
        assert "hidden_dim" in self.config, (
            "[ERROR] 'hidden_dim' must be specified in the config."
        )
        self.res: Tuple[int, int] = self.config["res"]
        self.n_channels: int = self.config["n_channels"]

        self.n_layers: int = self.config["n_layers"]
        self.latent_dim: int = self.config["latent_dim"]
        self.hidden_dim: int = self.config["hidden_dim"]

        self.dropout: float = self.config.get("dropout", 0.0)

        self.io_dim: int = int(np.prod(self.res))


class GANDiscriminatorConfig:
    def __init__(self, config: str | Dict) -> None:
        self.config: Dict[str, Any] = (
            config if isinstance(config, Dict) else load_yaml(config)
        )
        assert "n_layers" in self.config, (
            "[ERROR] 'n_layers' must be specified in the config."
        )
        assert "hidden_dim" in self.config, (
            "[ERROR] 'hidden_dim' must be specified in the config."
        )
        self.res: Tuple[int, int] = self.config["res"]
        self.n_channels: int = self.config["n_channels"]

        self.n_layers: int = self.config["n_layers"]
        self.hidden_dim: int = self.config["hidden_dim"]

        self.use_minibatch: bool = self.config.get("use_minibatch", True)
        self.minibatch_dim: int = self.config.get("minibatch_dim", 64)
        self.minibatch_inner_dim: int = self.config.get("minibatch_inner_dim", 16)
        self.dropout: float = self.config.get("dropout", 0.0)

        self.io_dim: int = int(np.prod(self.res))
