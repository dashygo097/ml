import os
from typing import Optional

import torch
from termcolor import colored
from torch import nn

from .quantizer import Quantizer


def pseudo_quantize_tensor(
    weight: torch.Tensor, nbits: int = 4, groupsize: int = -1
) -> torch.Tensor:
    ori_w_shape = weight.shape
    if groupsize != -1:
        weight = weight.reshape(-1, groupsize)

    wmax = weight.amax(dim=1, keepdim=True)
    wmin = weight.amin(dim=1, keepdim=True)

    maxint = 2**nbits - 1
    scales = (wmax - wmin).clamp(min=1e-5) / maxint
    zeros = (-torch.round(wmin / scales)).clamp_(0, maxint)
    weight = torch.clamp(torch.round(weight / scales) + zeros, 0, maxint)
    weight = (weight - zeros) * scales
    weight = weight.reshape(ori_w_shape)

    return weight


class AWQ(Quantizer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def save(self, save_dict: str, name: str = "awq_model") -> None:
        os.makedirs(save_dict, exist_ok=True)
        path = save_dict + "/" + name + ".pt"
        torch.save(self.model.state_dict(), path)
        print(
            "[INFO] Model saved at: "
            + colored(path, "light_green", attrs=["underline"])
            + "!"
        )

    def get_model_size(
        self, data_width: int = 16, group_size: int = -1, info: bool = False
    ):
        final_data_width = data_width
        if group_size != -1:
            final_data_width += (16 + 4) / group_size

        num_elements = 0
        for param in self.model.parameters():
            num_elements += param.numel()

        bits = num_elements * final_data_width
        if info:
            if bits < 8 * 1024 * 1024:
                print(
                    "[INFO] Model size: "
                    + colored(
                        f"{bits / (8 * 1024):.2f} KB", "light_blue", attrs=["bold"]
                    )
                )
            elif bits < 8 * 1024 * 1024 * 1024:
                print(
                    "[INFO] Model size: "
                    + colored(
                        f"{bits / (8 * 1024 * 1024):.2f} MB",
                        "light_blue",
                        attrs=["bold"],
                    )
                )
            else:
                print(
                    "[INFO] Model size: "
                    + colored(
                        f"{bits / (8 * 1024 * 1024 * 1024):.2f} GB",
                        "light_blue",
                        attrs=["bold"],
                    )
                )

        return bits

    def quantize(self, target: Optional[str] | type):
        if target is None:
            ...

        elif isinstance(target, str):
            ...

        elif isinstance(target, type):
            ...

    def _pseudo_quantize(self, nbits: int = 4, groupsize: int = -1) -> None:
        for name, param in self.model.named_modules():
            if isinstance(param, nn.Linear):
                param.weight.data = pseudo_quantize_tensor(
                    param.weight.data, nbits=nbits, groupsize=groupsize
                )
                self.replace(name, lambda: param)
