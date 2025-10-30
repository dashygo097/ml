import warnings
from typing import Any, Dict, List, Optional, Union

import brevitas.nn as qnn
import torch
from torch import nn

from ..quantizer import Quantizer

BREVITAS_QUANTABLE_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nn.RNN,
    nn.LSTM,
)


class BrevitasPTQ(Quantizer):
    def __init__(
        self,
        model: nn.Module,
        weight_bit_width: int,
        activation_bit_width: int,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(model)
        self.weight_bit_width = weight_bit_width
        self.activation_bit_width = activation_bit_width
        self.calibration_data = calibration_data
        self._quantized_modules: List[str] = []

    def quantize(self, target: Optional[Union[str, type]]) -> List[str]:
        if target is None:
            return self._quantize_all()
        elif isinstance(target, str):
            return self._quantize_by_name(target)
        elif isinstance(target, type):
            return self._quantize_by_type(target)
        return []

    def _quantize_all(self) -> List[str]:
        quantized = []
        for name, module in self.model.named_modules():
            if name and self._is_quantizable(module):  # Skip root module
                self._replace_with_brevitas_module(name, module)
                quantized.append(name)

        self._quantized_modules = quantized
        if self.calibration_data is not None:
            self._calibrate()

        return quantized

    def _quantize_by_name(self, target: str) -> List[str]:
        quantized = []

        for name, module in self.model.named_modules():
            # Match exact name or name ending with target
            if name == target or name.endswith("." + target):
                if self._is_quantizable(module):
                    self._replace_with_brevitas_module(name, module)
                    quantized.append(name)

        if quantized:
            self._quantized_modules.extend(quantized)
            if self.calibration_data is not None:
                self._calibrate()
            return quantized

        warnings.warn(f"No quantizable module found matching '{target}'", UserWarning)
        return []

    def _quantize_by_type(self, target: type) -> List[str]:
        quantized = []
        for name, module in self.model.named_modules():
            if name and isinstance(module, target) and self._is_quantizable(module):
                self._replace_with_brevitas_module(name, module)
                quantized.append(name)

        self._quantized_modules = quantized
        if self.calibration_data is not None:
            self._calibrate()

        return quantized

    def _is_quantizable(self, module: nn.Module) -> bool:
        return isinstance(module, BREVITAS_QUANTABLE_TYPES)

    def _replace_with_brevitas_module(self, name: str, module: nn.Module) -> None:
        brevitas_mapping = {
            nn.Conv1d: self._create_quant_conv1d,
            nn.Conv2d: self._create_quant_conv2d,
            nn.Conv3d: self._create_quant_conv3d,
            nn.Linear: self._create_quant_linear,
            nn.RNN: self._create_quant_rnn,
            nn.LSTM: self._create_quant_lstm,
        }

        module_type = type(module)
        if module_type not in brevitas_mapping:
            return

        quantized_module = brevitas_mapping[module_type](module)

        if quantized_module is None:
            return

        self._copy_weights(module, quantized_module)

        # Replace in model
        self._set_module_by_name(name, quantized_module)

    def _create_quant_linear(self, module: nn.Linear) -> Optional[qnn.QuantLinear]:
        try:
            return qnn.QuantLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                weight_bit_width=self.weight_bit_width,
                act_bit_width=self.activation_bit_width,
            )
        except Exception as e:
            warnings.warn(f"Failed to create QuantLinear: {e}", UserWarning)
            return None

    def _create_quant_conv2d(self, module: nn.Conv2d) -> Optional[qnn.QuantConv2d]:
        try:
            return qnn.QuantConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                weight_bit_width=self.weight_bit_width,
                act_bit_width=self.activation_bit_width,
            )
        except Exception as e:
            warnings.warn(f"Failed to create QuantConv2d: {e}", UserWarning)
            return None

    def _create_quant_conv1d(self, module: nn.Conv1d) -> Optional[qnn.QuantConv1d]:
        try:
            return qnn.QuantConv1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                weight_bit_width=self.weight_bit_width,
                act_bit_width=self.activation_bit_width,
            )
        except Exception as e:
            warnings.warn(f"Failed to create QuantConv1d: {e}", UserWarning)
            return None

    def _create_quant_conv3d(self, module: nn.Conv3d) -> Optional[qnn.QuantConv3d]:
        try:
            return qnn.QuantConv3d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                weight_bit_width=self.weight_bit_width,
                act_bit_width=self.activation_bit_width,
            )
        except Exception as e:
            warnings.warn(f"Failed to create QuantConv3d: {e}", UserWarning)
            return None

    def _create_quant_rnn(self, module: nn.RNN) -> Optional[qnn.QuantRNN]:
        try:
            return qnn.QuantRNN(
                input_size=module.input_size,
                hidden_size=module.hidden_size,
                num_layers=module.num_layers,
                bias=module.bias,
                batch_first=module.batch_first,
                dropout=module.dropout,
                bidirectional=module.bidirectional,
                weight_bit_width=self.weight_bit_width,
                act_bit_width=self.activation_bit_width,
            )
        except Exception as e:
            warnings.warn(f"Failed to create QuantRNN: {e}", UserWarning)
            return None

    def _create_quant_lstm(self, module: nn.LSTM) -> Optional[qnn.QuantLSTM]:
        try:
            return qnn.QuantLSTM(
                input_size=module.input_size,
                hidden_size=module.hidden_size,
                num_layers=module.num_layers,
                bias=module.bias,
                batch_first=module.batch_first,
                dropout=module.dropout,
                bidirectional=module.bidirectional,
                weight_bit_width=self.weight_bit_width,
                act_bit_width=self.activation_bit_width,
            )
        except Exception as e:
            warnings.warn(f"Failed to create QuantLSTM: {e}", UserWarning)
            return None

    def _copy_weights(self, original: nn.Module, quantized: nn.Module) -> None:
        try:
            if hasattr(original, "weight") and hasattr(quantized, "weight"):
                quantized.weight.data.copy_(original.weight.data)
            if hasattr(original, "bias") and original.bias is not None:
                if hasattr(quantized, "bias") and quantized.bias is not None:
                    quantized.bias.data.copy_(original.bias.data)
        except Exception as e:
            warnings.warn(f"Failed to copy weights: {e}", UserWarning)

    def _set_module_by_name(self, name: str, new_module: nn.Module) -> None:
        parts = name.split(".")
        if len(parts) == 1:
            setattr(self.model, name, new_module)
        else:
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

    def _calibrate(self) -> None:
        self.model.eval()
        with torch.no_grad():
            try:
                if isinstance(self.calibration_data, (list, tuple)):
                    for batch in self.calibration_data:
                        if isinstance(batch, dict):
                            self.model(**batch)
                        else:
                            self.model(batch)
                else:
                    self.model(self.calibration_data)
            except Exception as e:
                warnings.warn(f"Calibration failed: {e}", UserWarning)

    def set_calibration_data(self, calibration_data: torch.Tensor) -> None:
        self.calibration_data = calibration_data

    def get_quantized_modules(self) -> List[str]:
        return self._quantized_modules

    def list_quantizable_modules(self) -> List[str]:
        quantizable = []
        for name, module in self.model.named_modules():
            if name and self._is_quantizable(module):
                quantizable.append(name)
        return quantizable

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "weight_bit_width": self.weight_bit_width,
            "activation_bit_width": self.activation_bit_width,
            "quantized_modules": self._quantized_modules,
        }
