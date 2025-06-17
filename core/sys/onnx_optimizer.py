from .editor import Editor
import onnx
import onnxoptimizer
from typing import List, Optional
from termcolor import colored
from typing import Tuple
import torch
from torch import nn
import os


class OnnxOptimizer(Editor):
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.onnx_model = None

    def export(self, save_dir: str = "./checkpoints/onnx", name: str = "model") -> None:
        if self.onnx_model is None:
            raise ValueError(
                "[ERROR] ONNX model is not loaded. Please load an ONNX model first."
            )

        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/{name}" + ".onnx"
        onnx.save(self.onnx_model, path)
        print(
            colored(
                f"[INFO] Exported ONNX model to {path}", "light_green", attrs=["bold"]
            )
        )

    def optimize(
        self,
        passes: Optional[List[str]] = [
            "eliminate_identity",
            "eliminate_deadend",
            "eliminate_nop_dropout",
            "eliminate_nop_transpose",
        ],
    ) -> None:
        if self.onnx_model is None:
            raise ValueError(
                "[ERROR] ONNX model is not loaded. Please load an ONNX model first."
            )
        self.onnx_model = onnxoptimizer.optimize(self.onnx_model, passes)
        print(
            colored(
                "[INFO] ONNX model optimization completed.",
                color="blue",
            )
        )

    def load_onnx(self, onnx_model_path: str) -> None:
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Model file {onnx_model_path} does not exist.")
        self.onnx_model = onnx.load(onnx_model_path)
        print(
            colored(
                f"[INFO] Loaded ONNX model from {onnx_model_path}",
                "light_green",
                attrs=["bold"],
            )
        )

    def export_onnx(
        self,
        input_shape: Tuple[int, ...],
        save_dir: str = "./checkpoints/onnx",
        name: str = "model",
        opset: int = 11,
        device: str = "cpu",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        input_tensor = torch.randn(input_shape).to(device)
        path = f"{save_dir}/{name}" + ".onnx"
        self.model.to(device)
        self.model.eval()
        torch.onnx.export(
            self.model,
            input_tensor,
            path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            do_constant_folding=True,
        )
        print(
            colored(f"[INFO] Exported model to {path}", "light_green", attrs=["bold"])
        )
