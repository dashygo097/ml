import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Generic, TypeVar

import matplotlib.pyplot as plt
import torch
from termcolor import colored
from torch import nn
from tqdm.rich import tqdm

from .utils import load_yaml


class TrainerArgs:
    def __init__(self, path: str) -> None:
        self.args: Dict = load_yaml(path)
        self.device: str = self.args["device"]
        self.batch_size: int = self.args["batch_size"]
        self.n_epochs: int = self.args["n_epochs"]
        self.lr: float = self.args["lr"]

        self.is_shuffle: bool = self.args.get("is_shuffle", False)
        self.save_dict: str = self.args.get("save_dict", "./checkpoints")

        if "log_dict" not in self.args["info"].keys():
            self.log_dict: str = "./train_logs"
            if self.log_dict.endswith("/"):
                self.log_dict = self.log_dict[:-1]

        else:
            self.log_dict: str = self.args["info"]["log_dict"]

        self.is_draw: bool = self.args["info"]["is_draw"]


T_args = TypeVar("T_args", bound=TrainerArgs)
T_model = TypeVar("T_model", bound=nn.Module)


class Trainer(Generic[T_args, T_model], ABC):
    def __init__(
        self,
        model: T_model,
        dataset,
        criterion,
        args: T_args,
        optimizer=None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.args = args

        self.set_device(args.device)
        self.set_optimizer(optimizer)
        self.set_model(model)
        self.set_dataset(dataset)

        self.n_steps: int = 0
        self.n_epochs: int = 0
        self.logger: Dict = {}

    def set_device(self, device) -> None:
        if device is None:
            self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)

        elif isinstance(device, torch.device):
            self.device = device

        else:
            raise ValueError("Invalid device")

    def set_optimizer(self, optimizer) -> None:
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer

        else:
            raise ValueError("Invalid optimizer")

    def set_model(self, model) -> None:
        self.model = model.to(self.device)

    def set_dataset(self, dataset) -> None:
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.is_shuffle,
        )

    def save(self) -> None:
        os.makedirs(self.args.save_dict, exist_ok=True)
        path = self.args.save_dict + "/checkpoint_" + str(self.n_steps) + ".pt"
        torch.save(self.model.state_dict(), path)
        print(
            "[INFO] Model saved at: "
            + colored(path, "light_green", attrs=["underline"])
            + "!"
        )

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

    def save_log(self) -> None:
        os.makedirs(self.args.log_dict, exist_ok=True)
        path = self.args.log_dict + "/datalogs" + ".json"
        json.dump(self.logger, open(path, "w"))
        print(
            "[INFO] Log saved at: "
            + colored(path, "light_green", attrs=["underline"])
            + "!"
        )

    @abstractmethod
    def step(self, batch) -> Dict:
        # TODO: implement this function
        raise NotImplementedError

    def step_info(self, result: Dict) -> None:
        # TODO: implement this function
        pass

    def epoch_info(self) -> None:
        # TODO: implement this function
        pass

    def log2plot(self) -> None:
        # NOTE: You can reimplment this function if you want
        plt.style.use("ggplot")
        plot_titles = []
        datareg = {}
        for obj in self.logger.keys():
            for record in self.logger[obj]:
                plot_titles.append(record)
            break

        for record in plot_titles:
            datareg[record] = []

        for record in plot_titles:
            for obj in self.logger.keys():
                datareg[record].append(self.logger[obj][record])

        for data in datareg.keys():
            plt.plot(datareg[data], label=data)
            plt.title(f"{data} vs Epochs")
            plt.xticks(range(0, self.n_epochs, int(self.n_epochs / 10 + 0.5)))
            plt.xlabel("Epochs")
            plt.ylabel(data)
            plt.legend()
            plt.savefig(self.args.log_dict + "/" + data + ".png")
            plt.clf()

    def train(self) -> None:
        # NOTE: You can reimplment this function if you want
        for epoch in range(self.args.n_epochs):
            for i, batch in enumerate(
                tqdm(
                    self.data_loader,
                    total=len(self.data_loader),
                    desc=colored(f"epoch: {epoch}", "light_red", attrs=["bold"]),
                    leave=False,
                )
            ):
                step_result = self.step(batch)
                self.step_info(step_result)
                self.n_steps += 1

            self.epoch_info()
            self.n_epochs += 1

        self.save()
        self.save_log()
        if self.args.is_draw:
            self.log2plot()
