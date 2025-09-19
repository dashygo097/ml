from typing import Dict, Optional, Tuple

import torch
from torch import nn

from ....utils import load_yaml
from ...heads import ClassifyHead
from ..components import PatchEmbedding
from ..encoder import EncoderBlock


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
        if self.task not in ["none", "classification"]:
            raise ValueError(f"Unsupported task: {self.task}")
        elif self.task == "classification":
            assert "num_classes" in self.args, "num_classes must be specified"
            self.num_classes: int = self.args["num_classes"]

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
        for i in range(n_layers):
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
        self.post_layernorm = nn.LayerNorm(self.d_model, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        return self.post_layernorm(x)

    def from_hf(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Load Embedding
        self.embedding.proj.weight.data = state_dict.pop(
            "embeddings.patch_embeddings.projection.weight"
        )
        self.embedding.proj.bias.data = state_dict.pop(
            "embeddings.patch_embeddings.projection.bias"
        )
        self.embedding.cls_token.data = state_dict.pop("embeddings.cls_token")
        self.embedding.pos_embedding.data = state_dict.pop(
            "embeddings.position_embeddings"
        )
        # Load Attention
        for i in range(self.n_layers):
            query_weight = state_dict.pop(
                f"encoder.layer.{i}.attention.attention.query.weight"
            )
            query_bias = state_dict.pop(
                f"encoder.layer.{i}.attention.attention.query.bias"
            )
            key_weight = state_dict.pop(
                f"encoder.layer.{i}.attention.attention.key.weight"
            )
            key_bias = state_dict.pop(f"encoder.layer.{i}.attention.attention.key.bias")
            value_weight = state_dict.pop(
                f"encoder.layer.{i}.attention.attention.value.weight"
            )
            value_bias = state_dict.pop(
                f"encoder.layer.{i}.attention.attention.value.bias"
            )
            output_weight = state_dict.pop(
                f"encoder.layer.{i}.attention.output.dense.weight"
            )
            output_bias = state_dict.pop(
                f"encoder.layer.{i}.attention.output.dense.bias"
            )
            self.encoder[i].attn.W_qkv.weight.data = torch.cat(
                [query_weight, key_weight, value_weight], dim=0
            )
            self.encoder[i].attn.W_qkv.bias.data = torch.cat(
                [query_bias, key_bias, value_bias], dim=0
            )
            self.encoder[i].attn.W_o.weight.data = output_weight
            self.encoder[i].attn.W_o.bias.data = output_bias

        # Load FFN
        for i in range(self.n_layers):
            fc1_weight = state_dict.pop(f"encoder.layer.{i}.intermediate.dense.weight")
            fc1_bias = state_dict.pop(f"encoder.layer.{i}.intermediate.dense.bias")
            fc2_weight = state_dict.pop(f"encoder.layer.{i}.output.dense.weight")
            fc2_bias = state_dict.pop(f"encoder.layer.{i}.output.dense.bias")
            self.encoder[i].ffn.fc1.weight.data = fc1_weight
            self.encoder[i].ffn.fc1.bias.data = fc1_bias
            self.encoder[i].ffn.fc2.weight.data = fc2_weight
            self.encoder[i].ffn.fc2.bias.data = fc2_bias

        # Load LayerNorm
        for i in range(self.n_layers):
            ln1_weight = state_dict.pop(f"encoder.layer.{i}.layernorm_before.weight")
            ln1_bias = state_dict.pop(f"encoder.layer.{i}.layernorm_before.bias")
            ln2_weight = state_dict.pop(f"encoder.layer.{i}.layernorm_after.weight")
            ln2_bias = state_dict.pop(f"encoder.layer.{i}.layernorm_after.bias")

            self.encoder[i].addnorm1.norm_layer.weight.data = ln1_weight
            self.encoder[i].addnorm1.norm_layer.bias.data = ln1_bias
            self.encoder[i].addnorm2.norm_layer.weight.data = ln2_weight
            self.encoder[i].addnorm2.norm_layer.bias.data = ln2_bias

        # Load Post LayerNorm
        post_ln_weight = state_dict.pop("layernorm.weight")
        post_ln_bias = state_dict.pop("layernorm.bias")
        self.post_layernorm.weight.data = post_ln_weight
        self.post_layernorm.bias.data = post_ln_bias

        if len(state_dict) > 0:
            raise ValueError("Some weights are not loaded")
        else:
            print("All weights are loaded!")


class ViTRawModel(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.vit = ViTBackbone(
            embed_size=config.embed_size,
            patch_size=config.patch_size,
            n_heads=config.n_heads,
            n_layers=config.num_layers,
            res=config.res,
            in_channels=config.in_channels,
            d_inner=config.d_inner,
            d_model=config.d_model,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def from_hf(self, model: nn.Module) -> None:
        state_dict = model.state_dict()
        self.vit.from_hf(state_dict)


class VitClassifier(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.vit = ViTBackbone(
            embed_size=config.embed_size,
            patch_size=config.patch_size,
            n_heads=config.n_heads,
            n_layers=config.num_layers,
            res=config.res,
            in_channels=config.in_channels,
            d_inner=config.d_inner,
            d_model=config.d_model,
            dropout=config.dropout,
        )
        self.head = ClassifyHead(config.embed_size, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x)[:, 0, :])

    def from_hf(self, model: nn.Module) -> None:
        state_dict = model.state_dict()
        vit_state_dict = model.vit.state_dict()
        self.vit.from_hf(vit_state_dict)
        self.head.model.fc[0].weight.data = state_dict.pop("classifier.weight")
        self.head.model.fc[0].bias.data = state_dict.pop("classifier.bias")
