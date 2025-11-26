import json
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, overload

import matplotlib.pyplot as plt
import numpy as np
import torch
from ptflops import get_model_complexity_info
from termcolor import colored
from torch import nn


class Tracer:
    def __init__(self, model: nn.Module):
        self.model = model

    def load(self, path: str, device: str = "cpu") -> None:
        self.model.load_state_dict(torch.load(path, map_location=device))

    def save(self, path: str = "./results_dir/model.pth") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(
            "[INFO] Model saved at: "
            + colored(path, "light_green", attrs=["underline"])
            + "!"
        )

    def summary(self) -> None:
        print(self.model)
        self.numal(info=True)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def export_all_bins(
        self,
        output_dir: str = "./exported",
        dtype: str = "float32",
        prefix: str = "",
        include_metadata: bool = True,
        verbose: bool = True,
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
        }

        if dtype not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Choose from {list(dtype_map.keys())}"
            )

        np_dtype = dtype_map[dtype]
        state_dict = self.model.state_dict()

        metadata = {"dtype": dtype, "parameters": {}}
        file_paths = {}
        total_params = 0

        if verbose:
            print(
                colored(
                    "\n[TRACER] Exporting Model Weights to Binary",
                    "magenta",
                    attrs=["bold"],
                )
            )
            print(f"Output directory: {colored(output_dir, 'cyan')}")
            print(f"Data type: {colored(dtype, 'cyan')}")
            print("-" * 60)

        for name, tensor in state_dict.items():
            safe_name = name.replace(".", "_").replace("/", "_")
            filename = f"{prefix}_{safe_name}.bin" if prefix else f"{safe_name}.bin"
            filepath = os.path.join(output_dir, filename)

            data = tensor.detach().cpu().numpy()
            original_shape = data.shape
            original_dtype = str(data.dtype)

            data = data.astype(np_dtype)

            with open(filepath, "wb") as f:
                f.write(data.tobytes())

            file_paths[name] = filepath
            total_params += data.size

            metadata["parameters"][name] = {
                "filename": filename,
                "shape": list(original_shape),
                "original_dtype": original_dtype,
                "export_dtype": dtype,
                "size": int(data.size),
                "size_bytes": int(data.nbytes),
            }

            if verbose:
                shape_str = "x".join(map(str, original_shape))
                size_kb = data.nbytes / 1024
                print(f"✓ {name:40s} | {shape_str:20s} | {size_kb:8.2f} KB")

        if include_metadata:
            meta_file = f"{prefix}_metadata.json" if prefix else "metadata.json"
            metadata_path = os.path.join(output_dir, meta_file)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            if verbose:
                print("-" * 60)
                print(f"✓ Metadata: {colored(metadata_path, 'green')}")

        if verbose:
            total_mb = sum(m["size_bytes"] for m in metadata["parameters"].values()) / (
                1024 * 1024
            )
            print("-" * 60)
            print(f"Total parameters: {colored(f'{total_params:,}', 'yellow')}")
            print(f"Total size: {colored(f'{total_mb:.2f} MB', 'yellow')}")

        return file_paths

    def export_layer_bins(
        self,
        layer_name: str,
        output_dir: str = "./exported",
        dtype: str = "float32",
        save_text: bool = False,
        verbose: bool = True,
    ) -> Dict[str, str]:
        state_dict = self.model.state_dict()

        layer_params = {k: v for k, v in state_dict.items() if k.startswith(layer_name)}

        if not layer_params:
            raise ValueError(f"No parameters found for layer: {layer_name}")

        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(
                colored(
                    f"\n[TRACER] Exporting Layer: {layer_name}",
                    "magenta",
                    attrs=["bold"],
                )
            )
            print(f"Found {len(layer_params)} parameter tensors")
            print("-" * 60)

        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
        }
        np_dtype = dtype_map[dtype]

        file_paths = {}

        for name, tensor in layer_params.items():
            param_type = name.split(".")[-1]
            filename = f"{param_type}.bin"
            filepath = os.path.join(output_dir, filename)

            data = tensor.detach().cpu().numpy().astype(np_dtype)

            with open(filepath, "wb") as f:
                f.write(data.tobytes())

            file_paths[name] = filepath

            if save_text:
                text_path = filepath.replace(".bin", ".txt")
                np.savetxt(text_path, data.flatten(), fmt="%.8f")

            if verbose:
                shape_str = "x".join(map(str, data.shape))
                print(f"✓ {param_type:15s} | {shape_str:20s} | {filepath}")

        if verbose:
            print("-" * 60)
            print(
                f"Files saved in: {colored(output_dir, 'green', attrs=['underline'])}"
            )

        return file_paths

    def fuzzy_fetch(self, target: str) -> nn.Module:
        fetched_module = []
        for name, module in self.model.named_modules():
            if target in name:
                fetched_module.append(module)

        if not fetched_module:
            assert False, colored(
                f"Module with name '{target}' not found in the model.",
                "red",
                attrs=["bold"],
            )
        if len(fetched_module) > 1:
            assert False, colored(
                f"Multiple modules found with name '{target}': {fetched_module}",
                "red",
                attrs=["bold"],
            )
        return fetched_module[0]

    @overload
    def fetch(self, target: type) -> List[Tuple[str, nn.Module]]:
        return self.fetch(target=target)

    @overload
    def fetch(self, target: str) -> List[Tuple[str, nn.Module]]:
        return self.fetch(target=target)

    def numal(self, info: bool = False) -> int:
        num_params = 0
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                num_params += param.numel()

        if info:
            msg = "[TRACER] Model Parameter Count"
            print(colored("\n" + msg, "magenta", attrs=["bold"]))

            print(
                "Number of trainable parameters: "
                + colored(f"{num_params}", "light_yellow", attrs=["dark"])
            )
        return num_params

    def get_details(self, input_shape: Tuple, info: bool = False) -> Tuple[str, str]:
        warnings.warn(
            colored(
                "[WARN] Method `get_details` only returns the parmas of THE ORIGINAL MODEL`",
                "yellow",
                attrs=["bold"],
            )
        )
        macs, params = get_model_complexity_info(
            self.model,
            input_shape,
            as_strings=True,
            print_per_layer_stat=info,
            verbose=info,
        )

        if info:
            msg = "[TRACER] Model Complexity Details"
            print(colored("\n" + msg, "magenta", attrs=["bold"]))
            print("MACs: " + colored(f"{macs}", "light_yellow", attrs=["dark"]))
            print("Parameters: " + colored(f"{params}", "light_yellow", attrs=["dark"]))

        return str(macs), str(params)

    def sparsity_report(self, info: bool = True) -> Dict[str, float]:
        report = {}
        for name, param in self.model.named_parameters():
            total = param.numel()
            zeros = (param == 0).sum().item()
            sparsity = zeros / total
            report[name] = sparsity
            if info:
                print(f"{name}: {sparsity * 100:.2f}% sparse")
        return report

    def layer_output(
        self,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
        info: bool = False,
    ) -> Dict:
        records = defaultdict(list)
        handles = []
        curr_device = next(self.model.parameters()).device
        input = torch.randn(input_shape).to(device)

        def time_layer(name):
            def hook(module, input, output):
                records[name].append(output.shape)

            return hook

        self.model.to(device)
        self.model.eval()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module):
                handle = module.register_forward_hook(time_layer(name))
                handles.append(handle)
        with torch.no_grad():
            self.model(input)
        for handle in handles:
            handle.remove()
        if info:
            for name, t in records.items():
                if name:
                    print(f"{name}: {t}")
        self.model.to(curr_device)
        return records

    def layer_latency(
        self,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
        info: bool = False,
    ) -> Dict[str, float]:
        records = defaultdict(float)
        handles = []
        curr_device = next(self.model.parameters()).device
        input_tensor = torch.randn(input_shape).to(device)

        def sync():
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

        def time_layer(name):
            def pre_hook(module, input):
                sync()
                module._start_time = time.perf_counter()

            def post_hook(module, input, output):
                sync()
                end_time = time.perf_counter()
                if hasattr(module, "_start_time"):
                    records[name] += end_time - module._start_time
                    delattr(module, "_start_time")

            return pre_hook, post_hook

        self.model.to(device)
        self.model.eval()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module) and name:
                pre_hook, post_hook = time_layer(name)
                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(post_hook))

        with torch.no_grad():
            for _ in range(3):
                self.model(input_tensor)

        records.clear()

        with torch.no_grad():
            self.model(input_tensor)

        for handle in handles:
            handle.remove()

        if info:
            sorted_records = sorted(records.items(), key=lambda x: x[1], reverse=True)
            msg = "[TRACER] Layer-wise Latency Report"
            print(colored("\n" + msg, "magenta", attrs=["bold"]))
            print("\nLayer Latencies (sorted by time):")
            print("-" * 50)
            for name, t in sorted_records:
                print(f"{name}: {t * 1e6:.3f} µs")

            total_time = sum(records.values())
            print("-" * 50)
            print(
                colored(
                    "Total Layer Time Elapsed: "
                    + colored(
                        f"{total_time * 1e6:.3f}",
                        "light_yellow",
                        attrs=["dark"],
                    )
                    + " µs",
                )
            )

        self.model.to(curr_device)
        return dict(records)

    def io_latency(
        self,
        input_shape: Tuple[int, ...],
        warmup: int = 10,
        device: str = "cpu",
        info: bool = False,
    ) -> float:
        curr_device = next(self.model.parameters()).device
        input = torch.randn(input_shape).to(device)

        model = self.model.to(device)
        model.eval()

        times = []
        with torch.no_grad():
            for _ in range(warmup):
                model(input)

            start = time.perf_counter()
            model(input)
            end = time.perf_counter()

            times.append(end - start)

        perf_time = sum(times) / len(times)
        if info:
            msg = "[TRACER] IO Latency Report"
            print(colored("\n" + msg, "magenta", attrs=["bold"]))
            print(
                "IO Latency: "
                + colored(f"{perf_time * 1e3:.3f} ms", "light_yellow", attrs=["dark"])
            )

        model.to(curr_device)

        return perf_time * 1e3

    def draw_weight_distribution(self, bins=256, count_nonzero_only=False):
        fig, axes = plt.subplots(3, 3, figsize=(10, 6))
        axes = axes.ravel()
        plot_index = 0
        for name, param in self.model.named_parameters():
            if param.dim() > 1:
                ax = axes[plot_index]
                if count_nonzero_only:
                    param_cpu = param.detach().view(-1).cpu()
                    param_cpu = param_cpu[param_cpu != 0].view(-1)
                    ax.hist(param_cpu, bins=bins, density=True, color="blue", alpha=0.5)
                else:
                    param_cpu = param.detach().view(-1).cpu()
                    ax.hist(
                        param_cpu,
                        bins=bins,
                        density=True,
                        color="blue",
                        alpha=0.5,
                    )
                ax.set_xlabel(name)
                ax.set_ylabel("density")
                plot_index += 1
        fig.suptitle("Histogram of Weights")
        fig.tight_layout()
        fig.subplots_adjust(top=0.925)
        plt.show()

    def fetch(self, target: str | type) -> List[Tuple[str, nn.Module]]:
        modules = []
        if isinstance(target, type):
            for name, module in self.model.named_modules():
                if isinstance(module, target):
                    modules.append((name, module))
        elif isinstance(target, str):
            for name, module in self.model.named_modules():
                if name == target:
                    modules.append((name, module))

        if not modules:
            assert False, colored(
                f"Module with name or type '{target}' not found in the model.",
                "red",
                attrs=["bold"],
            )

        return modules
