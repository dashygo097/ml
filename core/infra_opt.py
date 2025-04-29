import coremltools as ct
import onnx
import onnxoptimizer
import torch
import torch.nn as nn


def ex2onnx(model: nn.Module, input_shape: tuple) -> None:
    model.eval()
    model_name = model.__class__.__name__.lower()
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model, (dummy_input,), f"{model_name}.onnx", verbose=True, opset_version=12
    )


def optim_onnx(onnx_model_path: str) -> None:
    onnx_model = onnx.load_model(onnx_model_path)
    passes = [
        "eliminate_identity",  # Remove redundant Identity nodes
        "eliminate_deadend",  # Remove nodes with no outputs
        "fuse_bn_into_conv",  # Fuse BatchNorm into Conv2d
    ]
    onnx_model = onnxoptimizer.optimize(onnx_model, passes)
    onnx.save_model(onnx_model, onnx_model_path)


def convert2ct(model: nn.Module):
    model_name = model.__class__.__name__.lower()
    dummy_input = torch.randn((1, 3, 512, 512)).to("mps")
    model.eval()
    traced_model = torch.jit.trace(model, dummy_input)
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
    )
    mlmodel.save(f"{model_name}.mlpackage")  # pyright: ignore
