import os
from typing import Optional, Tuple
from termcolor import colored

import torch
from torch import nn
from torch.export import export
import numpy as np

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


class TVMe2eOptimizer:
    def __init__(
        self,
        model: nn.Module,
        example_args: Tuple[torch.Tensor, ...],
    ) -> None:
        self.from_pytorch(model, example_args)

    def show(self) -> None:
        if self.mod is None:
            print(
                colored(
                    "[WARN] No model has been loaded. Please load a model first.",
                    color="yellow",
                    attrs=["bold"],
                )
            )
            return

        self.mod.show()

    def from_pytorch(
        self,
        model: nn.Module,
        example_args: Tuple[torch.Tensor, ...],
    ) -> None:
        self.model = model.eval()
        IS_IN_CI = os.getenv("CI", "") == "true"

        if not IS_IN_CI:
            print(
                colored(
                    "[INFO] Exporting PyTorch model to TVM...",
                    color="light_green",
                    attrs=["bold"],
                )
            )

            with torch.no_grad():
                exported_program = export(model, example_args)
                mod = from_exported_program(exported_program, keep_params_as_input=True)

            self.mod, self.params = relax.frontend.detach_params(mod)

    def auto_tune(
        self,
        target: str,
        total_trials: int = 1000,
        work_dir: str = "./tuning_logs",
        info: bool = False,
    ) -> None:
        target = tvm.target.Target(target)
        IS_IN_CI = os.getenv("CI", "") == "true"

        if not IS_IN_CI:
            if self.mod is None:
                raise ValueError(
                    colored(
                        "[ERROR] No model has been loaded. Please load a model first.",
                        color="red",
                        attrs=["bold"],
                    )
                )
            self.mod = relax.get_pipeline(
                "static_shape_tuning",
                target=target,
                total_trials=total_trials,
                work_dir=work_dir,
            )(self.mod)

            if info:
                self.mod.show()

    def optimize(
        self, input: np.ndarray, target: str, dtype: str = "float32"
    ) -> Optional[np.ndarray]:
        target = tvm.target.Target(target)
        target_output = None
        IS_IN_CI = os.getenv("CI", "") == "true"
        if not IS_IN_CI:
            ex = tvm.compile(self.mod, target)
            device = tvm.device(target)
            vm = relax.VirtualMachine(ex, device)

            target_data = tvm.nd.array(input.astype(dtype), device=device)
            target_params = [
                tvm.nd.array(p, device=device) for p in self.params["main"]
            ]
            target_output = vm["main"](target_data, *target_params).numpy()

        return target_output


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    optimizer = TVMe2eOptimizer(MLP(10, 20), (torch.randn(16, 10),))
    optimizer.show()

    optimizer.auto_tune("metal", total_trials=10, info=True)
    optimizer.optimize(np.random.randn(16, 10).astype("float32"), "metal")
