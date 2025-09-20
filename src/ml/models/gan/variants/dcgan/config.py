from typing import Any, Dict, Tuple

from .....utils import load_yaml


class DCGANGeneratorConfig:
    def __init__(self, config: str | Dict) -> None:
        self.config: Dict[str, Any] = (
            config if not isinstance(config, str) else load_yaml(config)
        )
        assert "latent_dim" in self.config, (
            "[ERROR] 'latent_dim' must be specified in the config."
        )
        assert "n_layers" in self.config, (
            "[ERROR] 'n_layers' must be specified in the config."
        )
        assert "hidden_dim" in self.config, (
            "[ERROR] 'hidden_dim' must be specified in the config."
        )
        self.res: Tuple[int, int] = self.config["res"]
        self.n_channels: int = self.config["n_channels"]

        self.latent_dim: int = self.config["latent_dim"]
        self.n_layers: int = self.config["n_layers"]
        self.hidden_dim: int = self.config["hidden_dim"]

        self.kernel_size: int = self.config.get("kernel_size", 3)
        self.dropout: float = self.config.get("dropout", 0.0)


class DCGANDiscriminatorConfig:
    def __init__(self, config: str | Dict) -> None:
        self.config: Dict[str, Any] = (
            config if not isinstance(config, str) else load_yaml(config)
        )
        assert "n_channels" in self.config, (
            "[ERROR] 'n_channels' must be specified in the config."
        )
        assert "n_layers" in self.config, (
            "[ERROR] 'n_layers' must be specified in the config."
        )
        self.res: Tuple[int, int] = self.config["res"]
        self.n_channels: int = self.config["n_channels"]

        self.n_layers: int = self.config["n_layers"]
        self.hidden_dim: int = self.config["hidden_dim"]

        self.kernel_size: int = self.config.get("kernel_size", 3)
        self.dropout: float = self.config.get("dropout", 0.0)
        self.use_minibatch: bool = self.config.get("use_minibatch", False)

        self.minibatch_dim: int = self.config.get("minibatch_dim", 64)
        self.minibatch_inner_dim: int = self.config.get("minibatch_in_features", 16)
        self.minibatch_out_features: int = self.config.get("minibatch_out_features", 64)
