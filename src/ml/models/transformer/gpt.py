from typing import Optional, OrderedDict

import torch
from torch import nn

from .attns import MulHeadLatentAttn
from .components import SwiGLUFFN
from .decoder import DecoderOnlyBlock


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        module_list = []

        for i in range(num_layers):
            module_list.append(
                (
                    f"blk_{i}",
                    DecoderOnlyBlock(
                        d_model=d_model,
                        n_heads=num_heads,
                        d_inner=d_model * 4,
                        dropout=dropout,
                        attn=MulHeadLatentAttn(
                            embed_size=d_model,
                            latent_dim=d_model // 3,
                            num_heads=num_heads,
                            dropout=dropout,
                        ),
                        ffn=SwiGLUFFN(
                            d_model=d_model,
                            d_inner=d_model * 4,
                            dropout=dropout,
                        ),
                    ),
                )
            )

        self.decoders = nn.Sequential(OrderedDict(module_list))
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        x = self.embedding(x)
        for blk in self.decoders:
            x = blk(x, mask=mask, is_causal=is_causal)

        return self.fc(self.dropout(x))
