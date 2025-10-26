from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torchao.quantization.quant_api import (Int4WeightOnlyConfig,
                                            Int8WeightOnlyConfig, quantize_)

from ..quantizer import Quantizer

_DtypeLike = Union[str, torch.dtype]


class AOPTQ(Quantizer):
    SUPPORTED_SCHEMES = {"weight_only"}
    SUPPORTED_DTYPES = {"int4", "i4", "int8", "i8"}

    def __init__(
        self,
        model: nn.Module,
        *,
        scheme: str = "weight_only",
        dtype: _DtypeLike = "int8",
        weight_dtype: Optional[_DtypeLike] = None,
        act_dtype: Optional[_DtypeLike] = None,
        group_size: Optional[int] = None,
        layout: Optional[object] = None,
        bit_width: Optional[int] = None,
        packing_bitwidth: Optional[int] = None,
        use_hqq: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        set_inductor_config: bool = True,
    ) -> None:
        super().__init__(model)

        self.scheme = self._validate_scheme(scheme)

        self.weight_dtype = self._normalize_dtype(weight_dtype or dtype)
        self.act_dtype = (
            self._normalize_dtype(act_dtype or dtype) if act_dtype else None
        )

        self.group_size = group_size
        self.layout = layout
        self.bit_width = bit_width
        self.packing_bitwidth = packing_bitwidth
        self.use_hqq = use_hqq
        self.device = device
        self.set_inductor_config = set_inductor_config

        self._validate_config()

        self._config = self._create_quantization_config()

    def quantize(self, target: Optional[Union[str, type]] = None) -> List[str]:
        predicate = self._create_predicate(target)

        selected = [
            name for name, mod in self.model.named_modules() if predicate(mod, name)
        ]

        if not selected:
            raise AssertionError(
                f"No modules matched target={target!r} for scheme={self.scheme}. "
                f"Available modules: {[name for name, _ in self.model.named_modules()]}"
            )

        quantize_(self.model, self._config, filter_fn=predicate, device=self.device)

        return selected

    def _validate_scheme(self, scheme: str) -> str:
        normalized = scheme.lower()
        if normalized not in self.SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unsupported scheme: {scheme}. "
                f"Supported schemes: {self.SUPPORTED_SCHEMES}"
            )
        return normalized

    def _normalize_dtype(self, dtype: _DtypeLike) -> str:
        dtype_str = str(dtype).lower().replace("torch.", "")

        dtype_map = {
            "i4": "int4",
            "i8": "int8",
            "int4": "int4",
            "int8": "int8",
        }

        if dtype_str not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Supported dtypes: {self.SUPPORTED_DTYPES}"
            )

        return dtype_map[dtype_str]

    def _validate_config(self) -> None:
        if self.weight_dtype == "int4" and self.group_size is None:
            raise ValueError(
                "INT4 quantization requires 'group_size' parameter. "
                "Common values: 32, 64, 128, 256"
            )

        if self.scheme == "dynamic" and self.act_dtype is None:
            raise ValueError(
                "Dynamic quantization scheme requires 'act_dtype' parameter"
            )

    def _create_quantization_config(self):
        if self.scheme == "weight_only":
            return self._create_weight_only_config()
        else:
            raise ValueError(f"Unsupported scheme: {self.scheme}")

    def _create_weight_only_config(self):
        if self.weight_dtype == "int8":
            return Int8WeightOnlyConfig(
                group_size=self.group_size,
                set_inductor_config=self.set_inductor_config,
            )

        elif self.weight_dtype == "int4":
            config_kwargs = {
                "group_size": self.group_size,
                "set_inductor_config": self.set_inductor_config,
            }

            if self.use_hqq is not None:
                config_kwargs["use_hqq"] = self.use_hqq

            return Int4WeightOnlyConfig(**config_kwargs)

        else:
            raise ValueError(
                f"Unsupported weight dtype for weight_only: {self.weight_dtype}"
            )

    def _create_predicate(
        self, target: Union[str, type, None]
    ) -> Callable[[nn.Module, str], bool]:
        if target is None:
            return lambda mod, name: True

        if isinstance(target, str):
            return lambda mod, name: name == target

        if isinstance(target, type):
            return lambda mod, name: isinstance(mod, target)

        raise TypeError(
            f"Target must be str, type, or None, got {type(target).__name__}"
        )
