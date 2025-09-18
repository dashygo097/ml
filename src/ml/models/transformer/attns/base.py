from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ..components import RoPE


@dataclass
class AttnInfraRecord:
    input_logits: torch.Tensor
    output_logits: Optional[torch.Tensor] = None
    attn_weights: Optional[torch.Tensor] = None
    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None


class AttnModel(ABC, nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        d_model: Optional[int] = None,
        bias: bool = False,
        enable_rope: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.d_model = d_model if d_model is not None else embed_size
        self.head_dim = self.d_model // self.n_heads
        self.bias = bias
        self.enable_rope = enable_rope
        self.dropout = dropout
        assert self.d_model % self.n_heads == 0, (
            f"[ERROR] d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        )

        if self.enable_rope:
            self.rope = RoPE(self.head_dim)
        self.out_dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor: ...

    @abstractmethod
    def qkv(
        self, x: torch.Tensor, pos: Optional[int] = None
    ) -> Tuple[torch.Tensor, ...]: ...

    def prompt(self, record: AttnInfraRecord) -> AttnInfraRecord: ...
    def infer(self, record: AttnInfraRecord) -> AttnInfraRecord: ...


class CrossAttnModel(ABC, nn.Module):
    def __init__(
        self,
        d_q: int,
        d_kv: int,
        n_heads: int,
        d_model: Optional[int] = None,
        bias: bool = False,
        enable_rope: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_size = d_q
        self.d_q = d_q
        self.d_kv = d_kv
        self.n_heads = n_heads
        self.d_model = d_model if d_model is not None else d_q
        self.head_dim = self.d_model // self.n_heads
        self.bias = bias
        self.dropout = dropout
        assert self.d_model % self.n_heads == 0, (
            f"[ERROR] d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        )

        if enable_rope:
            self.rope = RoPE(self.head_dim)
        self.out_dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...

    @abstractmethod
    def qkv(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
