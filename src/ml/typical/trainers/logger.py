import json
import os
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
from termcolor import colored


class TrainLogContent:
    def __init__(self) -> None:
        self.epoch: Dict[Any, Any] = {}
        self.step: Dict[Any, Any] = {}
        self.valid: Dict[Any, Any] = {}

    def __getitem__(self, key: str) -> Dict[Any, Any]:
        if key not in self.__dict__:
            raise KeyError(f"Key '{key}' not found in TrainLogContent.")

        try:
            return self.__dict__[key]
        except ValueError:
            print(
                colored(f"│ ERROR │ ", "red", attrs=["bold"]) +
                colored(f"Invalid keys in {key}. Ensure keys are integers.", "white")
            )
            return {}

    def __setitem__(self, key: str, value: Dict[Any, Any]) -> None:
        if key not in self.__dict__:
            raise KeyError(f"Key '{key}' not found in TrainLogContent.")
        self.__dict__[key] = value


class TrainLogger:
    def __init__(self, log_dict: str = "./train_logs") -> None:
        self._log_dict: str = log_dict
        self.content: TrainLogContent = TrainLogContent()

    @property
    def log_dict(self) -> str:
        """Get log directory path"""
        return self._log_dict

    @log_dict.setter
    def log_dict(self, value: str) -> None:
        """Set log directory path"""
        self._log_dict = value

    def save_log(self, info: bool = False) -> None:
        """Save training logs to JSON file"""
        os.makedirs(self.log_dict, exist_ok=True)
        path = os.path.join(self.log_dict, "datalogs.json")
        
        with open(path, "w") as f:
            json.dump(self.content.__dict__, f, indent=2)
        
        if info:
            print(
                colored(f"│ OK    │ ", "green", attrs=["bold"]) +
                colored(f"Log saved: {path}", "white", attrs=["bold"])
            )

    def op(self, key: str, func: Callable, index: Optional[int] = None) -> None:
        if index is None:
            index_key = str(len(self.content[key]) - 1)
        else:
            index_key = str(index)

        current_records = self.content[key]
        try:
            current_records[index_key] = func(current_records[index_key])
        except KeyError:
            current_records[index_key] = func({})

        self.content[key] = current_records

    def log(self, key: str, value: Dict[Any, Any], index: Optional[int] = None) -> None:
        if index is None:
            index_key = str(len(self.content[key]))
        else:
            index_key = str(index)
        
        current_records = self.content[key]
        current_records[index_key] = value
        self.content[key] = current_records

    def plot(self, key: str) -> None:
        os.makedirs(self.log_dict, exist_ok=True)
        plt.style.use("ggplot")
        elements: Dict[str, List] = {}

        records = self.content[key]

        try:
            n_begin = min(map(int, records.keys()))
            n_end = max(map(int, records.keys()))
        except ValueError:
            print(
                colored(f"│ ERROR │ ", "red", attrs=["bold"]) +
                colored(f"Invalid keys in {key}. Ensure keys are integers.", "white")
            )
            return

        for num, record in records.items():
            for label, value in record.items():
                if label not in elements:
                    elements[label] = []
                elements[label].append((int(num), value))

        for label in elements.keys():
            x = [item[0] for item in elements[label]]
            y = [item[1] for item in elements[label]]
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, linewidth=2.5, color="#FF6B6B", marker="o", markersize=4)
            plt.title(f"{label.upper()} vs {key.upper()}s", fontsize=14, fontweight="bold")
            plt.xticks(
                range(
                    n_begin,
                    n_end,
                    max(int((n_end - n_begin) / 10 + 0.5), 1),
                )
            )
            plt.xlabel(f"{key.upper()}", fontsize=12, fontweight="bold")
            plt.ylabel(label.upper(), fontsize=12, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(self._log_dict, f"{label}-{key}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.clf()
            plt.close()

        print(
            colored(f"│ OK    │ ", "green", attrs=["bold"]) +
            colored(f"Plots saved: {self._log_dict}", "white", attrs=["bold"])
        )
