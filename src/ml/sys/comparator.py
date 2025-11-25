from typing import Dict

import torch
from termcolor import colored
from torch import nn


class Comparator:
    def __init__(self, main_model: nn.Module, ref_model: nn.Module) -> None:
        self.main_model = main_model
        self.ref_model = ref_model

    def compare(
        self,
        input_tensor: torch.Tensor,
        device: str = "cpu",
        info: bool = True,
    ) -> Dict[str, float]:
        self.main_model = self.main_model.to(device)
        self.ref_model = self.ref_model.to(device)
        self.main_model.eval()
        self.ref_model.eval()

        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            main_output = self.main_model(input_tensor)
            ref_output = self.ref_model(input_tensor)

        main_flat = main_output.flatten()
        ref_flat = ref_output.flatten()

        mse = torch.mean((main_flat - ref_flat) ** 2).item()
        mae = torch.mean(torch.abs(main_flat - ref_flat)).item()
        rmse = torch.sqrt(torch.mean((main_flat - ref_flat) ** 2)).item()

        cos_sim = torch.nn.functional.cosine_similarity(
            main_flat.unsqueeze(0), ref_flat.unsqueeze(0)
        ).item()

        max_diff = torch.max(torch.abs(main_flat - ref_flat)).item()
        min_diff = torch.min(torch.abs(main_flat - ref_flat)).item()

        correlation = torch.corrcoef(torch.stack([main_flat, ref_flat]))[0, 1].item()

        main_mean = torch.mean(main_flat).item()
        main_std = torch.std(main_flat).item()
        ref_mean = torch.mean(ref_flat).item()
        ref_std = torch.std(ref_flat).item()

        comparison_result = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "cosine_similarity": cos_sim,
            "max_absolute_error": max_diff,
            "min_absolute_error": min_diff,
            "correlation": correlation,
            "main_output_shape": tuple(main_output.shape),
            "ref_output_shape": tuple(ref_output.shape),
            "main_output_mean": main_mean,
            "main_output_std": main_std,
            "ref_output_mean": ref_mean,
            "ref_output_std": ref_std,
            "main_output_min": torch.min(main_flat).item(),
            "main_output_max": torch.max(main_flat).item(),
            "ref_output_min": torch.min(ref_flat).item(),
            "ref_output_max": torch.max(ref_flat).item(),
        }

        if info:
            self._print_comparison(comparison_result)

        return comparison_result

    def _print_comparison(self, metrics: Dict[str, float]) -> None:
        msg = "[COMPARATOR] Comparing DUT and REF Models"
        print(colored("\n" + msg, "magenta", attrs=["bold"]))

        print(colored("Output Shapes:", "yellow", attrs=["bold"]))
        main_shape = str(metrics["main_output_shape"])
        ref_shape = str(metrics["ref_output_shape"])
        print("  Main Model: " + colored(main_shape, "light_green", attrs=["dark"]))
        print("  Ref Model:  " + colored(ref_shape, "light_yellow", attrs=["dark"]))
        print()

        print(colored("Error Metrics:", "yellow", attrs=["bold"]))
        mse_str = f"{metrics['mse']:.8f}"
        mae_str = f"{metrics['mae']:.8f}"
        rmse_str = f"{metrics['rmse']:.8f}"
        max_err_str = f"{metrics['max_absolute_error']:.8f}"
        min_err_str = f"{metrics['min_absolute_error']:.8f}"

        print(
            "  MSE (Mean Squared Error):      "
            + colored(mse_str, "light_green", attrs=["dark"])
        )
        print(
            "  MAE (Mean Absolute Error):     "
            + colored(mae_str, "light_green", attrs=["dark"])
        )
        print(
            "  RMSE (Root Mean Squared):      "
            + colored(rmse_str, "light_green", attrs=["dark"])
        )
        print(
            "  Max Absolute Error:            "
            + colored(max_err_str, "light_red", attrs=["dark"])
        )
        print(
            "  Min Absolute Error:            "
            + colored(min_err_str, "light_green", attrs=["dark"])
        )
        print()

        print(colored("Similarity Metrics:", "yellow", attrs=["bold"]))
        cos_sim_str = f"{metrics['cosine_similarity']:.6f}"
        corr_str = f"{metrics['correlation']:.6f}"
        print(
            "  Cosine Similarity:             "
            + colored(cos_sim_str, "light_green", attrs=["dark"])
        )
        print(
            "  Correlation:                   "
            + colored(corr_str, "light_green", attrs=["dark"])
        )
        print()

        print(colored("Main Model Statistics:", "yellow", attrs=["bold"]))
        main_mean_str = f"{metrics['main_output_mean']:.8f}"
        main_std_str = f"{metrics['main_output_std']:.8f}"
        main_min_str = f"{metrics['main_output_min']:.8f}"
        main_max_str = f"{metrics['main_output_max']:.8f}"

        print(
            "  Mean:                          "
            + colored(main_mean_str, "light_green", attrs=["dark"])
        )
        print(
            "  Std Dev:                       "
            + colored(main_std_str, "light_green", attrs=["dark"])
        )
        print(
            "  Min:                           "
            + colored(main_min_str, "light_green", attrs=["dark"])
        )
        print(
            "  Max:                           "
            + colored(main_max_str, "light_green", attrs=["dark"])
        )
        print()

        print(colored("Reference Model Statistics:", "yellow", attrs=["bold"]))
        ref_mean_str = f"{metrics['ref_output_mean']:.8f}"
        ref_std_str = f"{metrics['ref_output_std']:.8f}"
        ref_min_str = f"{metrics['ref_output_min']:.8f}"
        ref_max_str = f"{metrics['ref_output_max']:.8f}"

        print(
            "  Mean:                          "
            + colored(ref_mean_str, "light_yellow", attrs=["dark"])
        )
        print(
            "  Std Dev:                       "
            + colored(ref_std_str, "light_yellow", attrs=["dark"])
        )
        print(
            "  Min:                           "
            + colored(ref_min_str, "light_yellow", attrs=["dark"])
        )
        print(
            "  Max:                           "
            + colored(ref_max_str, "light_yellow", attrs=["dark"])
        )
        print()
