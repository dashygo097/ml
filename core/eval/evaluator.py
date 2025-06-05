import json
import os
from abc import ABC, abstractmethod

from termcolor import colored


class Evaluator(ABC):
    def __init__(self, device: str = "cpu") -> None:
        self.set_device(device)
        self.logger = {}

    def set_device(self, device: str):
        self.device = device

    @abstractmethod
    def bench(self, benchmark: str, log_dict: str = "eval_logs") -> None: ...

    def save_log(self, log_dict: str, info: bool = False) -> None:
        os.makedirs(log_dict, exist_ok=True)
        path = log_dict + "/datalogs" + ".json"
        json.dump(self.logger, open(path, "w"))
        if info:
            print(
                "[INFO] Log saved at: "
                + colored(path, "light_green", attrs=["underline"])
                + "!"
            )
