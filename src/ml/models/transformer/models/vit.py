from typing import Dict, Optional, OrderedDict, Tuple

import torch
from torch import nn

from ....utils import load_yaml
from ..components import PatchEmbedding
from ..encoder import EncoderBlock


class ViTConfig:
    def __init__(self, path_or_dict: str | Dict) -> None:
        self.args: Dict = (
            load_yaml(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
        )
        assert hasattr(self.args, "embed_size"), "embed_size is required"
        assert hasattr(self.args, "patch_size"), "patch_size is required"
        assert hasattr(self.args, "n_heads"), "n_heads is required"
        assert hasattr(self.args, "num_layers"), "num_layers is required"
        assert hasattr(self.args, "res"), "res is required"
        assert hasattr(self.args, "in_channels"), "in_channels is required"

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


class ViTBackbone(nn.Module):
    def __init__(
        self,
        embed_size: int,
        patch_size: int,
        n_heads: int,
        num_layers: int,
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
        self.num_layers = num_layers
        self.d_model = d_model if d_model is not None else embed_size
        self.d_inner = d_inner if d_inner is not None else 4 * self.d_model
        self.bias = True
        self.dropout = dropout

        # Image parameters
        self.res = res
        self.in_channels = in_channels

        self.embedding = PatchEmbedding(embed_size, res, patch_size, in_channels)

        module_list = []
        for i in range(num_layers):
            module_list.extend(
                [
                    (
                        f"blk_{i}",
                        EncoderBlock(
                            embed_size,
                            n_heads,
                            self.d_inner,
                            d_model=self.d_model,
                            norm=nn.LayerNorm(embed_size, eps=1e-12),
                            bias=True,
                            enable_rope=False,
                            postnorm=False,
                            dropout=dropout,
                        ),
                    ),
                ]
            )

        self.encoder = nn.Sequential(OrderedDict(module_list))
        self.post_layernorm = nn.LayerNorm(self.d_model, eps=1e-12)
        self.fc = nn.Linear(768, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        return self.post_layernorm(x)
