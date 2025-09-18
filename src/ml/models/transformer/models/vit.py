from typing import Optional, OrderedDict, Tuple

from torch import nn

from ...heads import ClassifyHead
from ..components import PatchEmbedding
from ..encoder import EncoderBlock


class ViT(nn.Module):
    def __init__(
        self,
        embed_size: int,
        patch_size: int,
        n_heads: int,
        num_layers: int,
        res: Tuple[int, int],
        in_channels: int,
        num_classes: int,
        d_inner: Optional[int] = None,
        d_model: Optional[int] = None,
        bias: bool = True,
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
        self.num_classes = num_classes

        self.embedding = PatchEmbedding(res, patch_size, in_channels, embed_size)

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
                            dropout=dropout,
                        ),
                    ),
                ]
            )

        self.encoder = nn.Sequential(OrderedDict(module_list))
        self.fc = ClassifyHead(self.d_model, num_classes)
