from typing import Any, Dict, List, Tuple

from .....utils import load_yaml


class ViTConfig:
    def __init__(self, path_or_dict: str | Dict) -> None:
        self.args: Dict = (
            load_yaml(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
        )
        assert "embed_size" in self.args, "embed_size must be specified"
        assert "patch_size" in self.args, "patch_size must be specified"
        assert "n_heads" in self.args, "n_heads must be specified"
        assert "num_layers" in self.args, "num_layers must be specified"
        assert "res" in self.args, "res must be specified"
        assert "in_channels" in self.args, "in_channels must be specified"

        self.task: str = self.args.get("task", "none")
        if self.task not in [
            "none",
            "classification",
            "change_detection",
            "depth_estimation"
        ]:
            raise ValueError(f"Unsupported task: {self.task}")
        elif self.task == "classification":
            assert "num_classes" in self.args, "num_classes must be specified"
            assert "head_type" in self.args, "head_type must be specified"
            self.num_classes: int = self.args["num_classes"]
            self.head_type: str = self.args["head_type"]
            self.head_args: Dict[str, Any] = self.args.get("head", {})
        elif self.task == "change_detection":
            assert "num_classes" in self.args, "num_classes must be specified"
            assert "head_type" in self.args, "head_type must be specified"
            self.num_classes: int = self.args["num_classes"]
            self.head_type: str = self.args["head_type"]
            self.head_args: Dict[str, Any] = self.args.get("head", {})
        elif self.task == "depth_estimation":
            assert "head_type" in self.args, "head_type must be specified"
            self.head_type: str = self.args["head_type"]
            self.head_args: Dict[str, Any] = self.args.get("head", {})

        self.embed_size: int = self.args["embed_size"]
        self.patch_size: int = self.args["patch_size"]
        self.n_heads: int = self.args["n_heads"]
        self.num_layers: int = self.args["num_layers"]
        self.res: Tuple[int, int] = tuple(self.args["res"])
        self.in_channels: int = self.args["in_channels"]

        if hasattr(self.args, "d_model"):
            self.d_model: int = self.args["d_model"]
        else:
            self.d_model: int = self.embed_size
        if hasattr(self.args, "d_inner"):
            self.d_inner: int = self.args["d_inner"]
        else:
            self.d_inner: int = 4 * self.d_model
        self.dropout: float = self.args.get("dropout", 0.0)

        if self.task == "classification" and self.head_type == "mlp":
            self.head_hidden_features: List[int] = self.head_args.get(
                "hidden_features", []
            )

        elif self.task == "change_detection" and self.head_type == "cnn":
            self.head_hidden_features: List[int] = self.head_args.get(
                "hidden_features", []
            )
            self.head_kernel_sizes: int | List[int] = self.head_args.get(
                "kernel_sizes", 3
            )

        elif self.task == "depth_estimation" and self.head_type == "cnn":
            self.head_hidden_features: List[int] = self.head_args.get(
                "hidden_features", []
            )
