import os
from typing import Optional, Tuple, Dict
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
        model: Optional[nn.Module] = None,
        example_args: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> None:
        self.model: Optional[nn.Module] = None

        self.mod: Optional[tvm.IRModule] = None
        self.params: Optional[Dict] = None

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

    def show_main(self) -> None:
        if self.mod is None:
            print(
                colored(
                    "[WARN] No model has been loaded. Please load a model first.",
                    color="yellow",
                    attrs=["bold"],
                )
            )
            return
        main_func = self.mod["main"]
        print(colored("[INFO] Main function:", color="light_blue", attrs=["bold"]))
        print(main_func)

    def from_pytorch(
        self,
        model: Optional[nn.Module],
        example_args: Optional[Tuple[torch.Tensor, ...]],
    ) -> None:
        if model is None:
            self.model = None
            return
        else:
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
                    if example_args is None:
                        raise ValueError(
                            colored(
                                "[ERROR] Example arguments must be provided for model export.",
                                color="red",
                                attrs=["bold"],
                            )
                        )
                    exported_program = export(model, example_args, strict=False)
                    mod = from_exported_program(
                        exported_program, keep_params_as_input=True
                    )

                self.mod, self.params = relax.frontend.detach_params(mod)

    def apply_database(
        self,
        target: str,
        totol_trials: int = 6000,
        work_dir: str = "./tuning_logs",
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
                total_trials=totol_trials,
                work_dir=work_dir,
            )(self.mod)
            self.mod["main"].show()

    def optimize(
        self, input_shape: Tuple[int, ...], target: str
    ) -> Optional[np.ndarray]:
        target = tvm.target.Target(target)
        target_output = None
        IS_IN_CI = os.getenv("CI", "") == "true"
        if not IS_IN_CI:
            ex = tvm.compile(self.mod, target=target)
            dev = tvm.device(target)
            vm = relax.VirtualMachine(ex, dev)

            if self.params is None:
                raise ValueError(
                    colored(
                        "[ERROR] Parameters are not set. Please load a model first.",
                        color="red",
                        attrs=["bold"],
                    )
                )
            target_data = tvm.nd.array(input_shape, dev)
            target_params = [tvm.nd.array(p, dev) for p in self.params["main"]]
            target_output = vm["main"](target_data, *target_params).numpy()

        return target_output


if __name__ == "__main__":
    optimizer = TVMe2eOptimizer()
    optimizer.from_pytorch(nn.Linear(10, 20), (torch.randn(16, 10),))
    optimizer.apply_database("metal")
