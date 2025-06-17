import os
from typing import Dict, List, Optional
from termcolor import colored
import matplotlib.pyplot as plt
import json


class TrainLogContent:
    epoch: Dict = {}
    step: Dict = {}
    valid: Dict = {}

    def __getitem__(self, key: str) -> Dict:
        if key not in self.__dict__:
            raise KeyError(f"Key '{key}' not found in TrainLogContent.")
        return dict(sorted(self.__dict__[key].items(), key=lambda x: int(x[0])))


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

    def log(self, key: str, value: Dict, index: Optional[int] = None) -> None:
        if index is None:
            index_key = str(len(self.content[key]))
        else:
            index_key = str(index)
        current_records = self.content[key]
        current_records[index_key] = value
        self.content.__dict__[key] = current_records

    def plot(self, key: str) -> None:
        # NOTE: Can be reimpled this function if want to

        os.makedirs(self.log_dict, exist_ok=True)
        plt.style.use("ggplot")
        elements: Dict[str, List] = {}

        records = self.content[key]

        n_records = len(records.keys())

        for record in records.values():
            for label, value in record.items():
                if label not in elements:
                    elements[label] = []
                elements[label].append(value)

        for label in elements.keys():
            plt.plot(elements[label], label=label)
            plt.title(f"{label} vs {key}s")
            plt.xticks(
                range(
                    0,
                    n_records,
                    max(int(n_records / 10 + 0.5), 1),
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
