import torch
from typing import Optional, List
from torch import nn
from .cross_attn import MulHeadCrossAttn


class HierarchicalAttnFusion(nn.Module):
    def __init__(
        self,
        embed_size: int,
        ds_q: List[int],
        d_kv: int,
        n_heads: int,
        d_model: Optional[int] = None,
    ) -> None:
        self.embed_size = embed_size
        self.ds_q = ds_q
        self.d_kv = d_kv
        self.n_heads = n_heads
        self.d_model = d_model if d_model is not None else embed_size
        self.head_dim = self.d_model // n_heads

        self.W_q = nn.ModuleList(
            [nn.Conv2d(q_dim, self.d_model, kernel_size=1) for q_dim in ds_q]
        )
        self.W_kv = nn.Linear(self.d_kv, self.d_model * 2, bias=False)

        self.cross_attns = nn.ModuleList([])

        self.W_o = nn.Linear(self.d_model * len(ds_q), embed_size)

    def forward(self, feats: List[torch.Tensor], kv: torch.Tensor) -> torch.Tensor:
        outputs = []
        ...
