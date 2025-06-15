from typing import Callable, List, overload

from torch import nn

from .tracer import Tracer


class Editor(Tracer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    @overload
    def replace(
        self, target: str, new_constructor: Callable[[], nn.Module]
    ) -> List[str]:
        return self.replace(target, new_constructor)

    @overload
    def replace(
        self, target: type, new_constructor: Callable[[], nn.Module]
    ) -> List[str]:
        return self.replace(target, new_constructor)

    @overload
    def freeze(self, target: str) -> None:
        self.freeze(target)

    @overload
    def freeze(self, target: type) -> None:
        self.freeze(target)

    def freeze(self, target: str | type) -> None:
        if isinstance(target, str):
            for name, module in self.model.named_modules():
                if name == target:
                    for param in module.parameters():
                        param.requires_grad = False
                    break
        elif isinstance(target, type):
            for name, module in self.model.named_modules():
                if isinstance(module, target):
                    for param in module.parameters():
                        param.requires_grad = False

    def replace(
        self, target: str | type, new_constructor: Callable[[], nn.Module]
    ) -> List[str]:
        replaced_modules = []
        if isinstance(target, str):
            for name, module in self.model.named_modules():
                if name == target:
                    new_module = new_constructor()
                    setattr(self.model, name, new_module)
                    replaced_modules.append(name)
                    break

        elif isinstance(target, type):
            for name, module in self.model.named_modules():
                if isinstance(module, target):
                    new_module = new_constructor()
                    setattr(self.model, name, new_module)
                    replaced_modules.append(name)

        return replaced_modules
