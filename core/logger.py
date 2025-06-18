import os
from typing import Callable
from typing import Dict, List, Optional
from matplotlib.lines import lineMarkers
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
import json


class TrainLogContent:
    def __init__(self) -> None:
        self.epoch: Dict = {}
        self.step: Dict = {}
        self.valid: Dict = {}

    def __getitem__(self, key: str) -> Dict:
        if key not in self.__dict__:
            raise KeyError(f"[ERROR] Key '{key}' not found in TrainLogContent.")

        return dict(sorted(self.__dict__[key].items(), key=lambda x: x))

    def __setitem__(self, key: str, value: Dict) -> None:
        if key not in self.__dict__:
            raise KeyError(f"[ERROR] Key '{key}' not found in TrainLogContent.")
        self.__dict__[key] = value


class TrainLogger:
    def __init__(self, log_dict: str = "./train_logs") -> None:
        self._log_dict: str = log_dict
        self.content: TrainLogContent = TrainLogContent()

    @property
    def log_dict(self) -> str:
        return self._log_dict

    @log_dict.setter
    def log_dict(self, value: str) -> None:
        self._log_dict = value

    def save_log(self, info: bool = False) -> None:
        os.makedirs(self.log_dict, exist_ok=True)
        path = self.log_dict + "/datalogs" + ".json"
        json.dump(self.content.__dict__, open(path, "w"))
        if info:
            print(
                "[INFO] Log saved at: "
                + colored(path, "light_green", attrs=["underline"])
                + "!"
            )

    def op(self, key: str, func: Callable, index: Optional[int] = None) -> None:
        if index is None:
            index_key = str(len(self.content[key]) - 1)
        else:
            index_key = str(index)
        current_records = self.content[key]
        current_records[index_key] = func(current_records[index_key])
        self.content[key] = current_records

    def log(self, key: str, value: Dict, index: Optional[int] = None) -> None:
        if index is None:
            index_key = str(len(self.content[key]))
        else:
            index_key = str(index)
        current_records = self.content[key]
        current_records[index_key] = value
        self.content[key] = current_records

    def plot(self, key: str) -> None:
        # NOTE: Can be reimpled this function if want to

        os.makedirs(self.log_dict, exist_ok=True)
        plt.style.use("ggplot")
        elements: Dict[str, List] = {}

        records = self.content[key]

        n_begin = min(map(int, records.keys()))
        n_end = max(map(int, records.keys()))

        for num, record in records.items():
            for label, value in record.items():
                if label not in elements:
                    elements[label] = []
                elements[label].append((int(num), value))

        for label in elements.keys():
            x = [item[0] for item in elements[label]]
            y = [item[1] for item in elements[label]]
            plt.plot(x, y, label=label)
            plt.title(f"{label} vs {key}s")
            plt.xticks(
                range(
                    n_begin,
                    n_end,
                    max(int((n_end - n_begin) / 10 + 0.5), 1),
                )
            )
            plt.xlabel(f"{key}")
            plt.ylabel(label)
            plt.legend()
            plt.savefig(self._log_dict + "/" + label + "-" + f"{key}.png")
            plt.clf()

        print(
            colored(
                f"[INFO] Plots saved at: {self._log_dict}!",
                "light_yellow",
                attrs=["underline"],
            )
        )
