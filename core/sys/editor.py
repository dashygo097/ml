import os
from typing import Callable, List, Optional, Tuple, overload

import torch
from termcolor import colored
from torch import nn

from .tracer import Tracer


class Editor(Tracer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def save(self, save_dict: str, name: str = "edited_model") -> None:
        os.makedirs(save_dict, exist_ok=True)
        path = save_dict + "/" + name + ".pt"
        torch.save(self.model.state_dict(), path)
        print(
            "[INFO] Model saved at: "
            + colored(path, "light_green", attrs=["underline"])
            + "!"
        )

    @overload
    def replace(
        self,
        target: str,
        new_constructor: Callable[[], nn.Module],
    ) -> List[str]:
        return self.replace(target, new_constructor)

    @overload
    def replace(
        self,
        target: type,
        new_constructor: Callable[[], nn.Module],
    ) -> List[str]:
        return self.replace(target, new_constructor)

    @overload
    def freeze(self, target: str) -> List[str]:
        self.freeze(target)

    @overload
    def freeze(self, target: type) -> List[str]:
        self.freeze(target)

    def freeze(self, target: str | type) -> List[str]:
        frozen_modules = []
        if isinstance(target, str):
            for name, module in self.model.named_modules():
                if name == target:
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen_modules.append(name)
                    break
        elif isinstance(target, type):
            for name, module in self.model.named_modules():
                if isinstance(module, target):
                    for param in module.parameters():
                        param.requires_grad = False

                    frozen_modules.append(name)

        if not frozen_modules:
            assert False, colored(
                f"[ERROR] No module of name or type {target} found in the model.",
                "red",
                attrs=["bold"],
            )
        return frozen_modules

    def replace(
        self,
        target: str | type,
        new_constructor: Callable[[], nn.Module | nn.Module],
    ) -> List[str]:
        replaced_modules = []
        if isinstance(target, str):
            for name, module in self.model.named_modules():
                if name == target:
                    old_name, parent_module = self._get_parent_module(name)
                    new_module = new_constructor()

                    setattr(parent_module, old_name, new_module)
                    replaced_modules.append(name)
                    break

        elif isinstance(target, type):
            for name, module in self.model.named_modules():
                if isinstance(module, target):
                    new_module = new_constructor()
                    setattr(self.model, name, new_module)
                    replaced_modules.append(name)

        if not replaced_modules:
            assert False, colored(
                f"[ERROR] No module of type {target} found in the model.",
                "red",
                attrs=["bold"],
            )

        return replaced_modules

    def _get_parent_module(self, name_path: str) -> Tuple[str, Optional[nn.Module]]:
        parts = name_path.split(".")
        submod = None
        for part in parts[:-1]:
            submod = getattr(self.model, part)
        return parts[-1], submod
