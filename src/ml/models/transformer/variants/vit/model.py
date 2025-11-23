from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .....utils import load_yaml
from ...components import PatchEmbedding
from ...encoder import EncoderBlock


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
            "obb_detection",
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
        elif self.task == "obb_detection":
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

        elif self.task == "obb_detection" and self.head_type == "detr_obb":
            self.head_num_queries: int = self.head_args.get("num_queries", 100)
            self.head_n_heads: int = self.head_args.get("n_heads", 8)
            self.head_d_model: int = self.head_args.get("d_model", self.d_model)
            self.head_n_layers: int = self.head_args.get("n_layers", 6)

        elif self.task == "change_detection" and self.head_type == "cnn_cd":
            self.head_hidden_features: List[int] = self.head_args.get(
                "hidden_features", []
            )
            self.head_kernel_sizes: int | List[int] = self.head_args.get(
                "kernel_sizes", 3
            )
        elif self.task == "change_detection" and self.head_type == "fpn_cd":
            raise NotImplementedError(
                "FPN-based change detection head is not implemented yet."
            )

        elif self.task == "depth_estimation" and self.head_type == "base_decoder":
            self.head_hidden_features: List[int] = self.head_args.get(
                "hidden_features", []
            )

class ViTBackbone(nn.Module):
    def __init__(
        self,
        embed_size: int,
        patch_size: int,
        n_heads: int,
        n_layers: int,
        res: Tuple[int, int],
        in_channels: int,
        d_inner: Optional[int] = None,
        d_model: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Model parameters
        self.embed_size = embed_size
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_model = d_model if d_model is not None else embed_size
        self.d_inner = d_inner if d_inner is not None else 4 * self.d_model
        self.bias = True
        self.dropout = dropout

        # Image parameters
        self.res = res
        self.in_channels = in_channels

        self.embedding = PatchEmbedding(embed_size, res, patch_size, in_channels)

        module_list = []
        for _ in range(n_layers):
            module_list.extend(
                [
                    EncoderBlock(
                        embed_size,
                        n_heads,
                        self.d_inner,
                        d_model=self.d_model,
                        norm1=nn.LayerNorm(embed_size, eps=1e-12),
                        norm2=nn.LayerNorm(embed_size, eps=1e-12),
                        bias=True,
                        enable_rope=False,
                        postnorm=False,
                        dropout=dropout,
                    )
                ]
            )

        self.encoder = nn.Sequential(*module_list)
        self.post_norm = nn.LayerNorm(self.d_model, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        return self.post_norm(x)
