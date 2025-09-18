from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


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
        d_model: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.d_model: int = d_model if d_model is not None else embed_size
        self.bias = bias
        self.dropout = dropout

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
        d_model: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_q = d_q
        self.d_kv = d_kv
        self.embed_size = d_q
        self.d_model = d_model if d_model is not None else d_q
        self.bias = bias
        self.dropout = dropout

        self.out_dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...

    @abstractmethod
    def qkv(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
