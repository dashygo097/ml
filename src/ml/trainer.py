import os
import threading
from abc import ABC, abstractmethod
from typing import Dict, Generic, List, TypeVar

import torch
from termcolor import colored
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from .logger import TrainLogger
from .utils import load_yaml


class TrainArgs:
    def __init__(self, path_or_dict: str | Dict) -> None:
        self.args: Dict = (
            load_yaml(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
        )
        self.device: str = self.args["device"]
        self.batch_size: int = self.args["batch_size"]
        self.n_epochs: int = self.args["n_epochs"]
        self.num_workers: int = self.args.get("num_workers", 2)
        self.lr: float = self.args["lr"]
        self.weight_decay: float = self.args.get("weight_decay", 0.0)

        self.is_shuffle: bool = self.args.get("is_shuffle", False)
        self.save_dict: str = self.args.get("save_dict", "./checkpoints")

        self.epochs_per_validation: int = self.args.get("epochs_per_validation", 1)

        if "log_dict" not in self.args["info"].keys():
            self.log_dict: str = "./train_logs"
            if self.log_dict.endswith("/"):
                self.log_dict = self.log_dict[:-1]

        else:
            self.log_dict: str = self.args["info"]["log_dict"]

        self.drawing_list: List[str] = self.args["info"].get("drawing_list", [])
        self.is_draw: bool = self.args["info"].get("is_draw", False)


T_args = TypeVar("T_args", bound=TrainArgs)
T_model = TypeVar("T_model", bound=nn.Module)


class Trainer(Generic[T_args, T_model], ABC):
    def __init__(
        self,
        model: T_model,
        dataset,
        criterion,
        args: T_args,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.args = args

        self.set_device(args.device)
        self.set_model(model)
        self.set_dataset(dataset)
        self.set_optimizer(optimizer)
        self.set_schedulers(scheduler)
        self.set_valid_ds(valid_ds)
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        if self.scaler is None:
            print(
                colored(
                    "[WARN] CUDA is NOT available, scalar won't be used.",
                    "yellow",
                )
            )

        self.n_steps: int = 0
        self.n_epochs: int = 0
        self.logger: TrainLogger = TrainLogger(self.args.log_dict)

    def set_device(self, device) -> None:
        if device is None:
            self.device = torch.device(
                "mps"
                if torch.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
            self.args.device = device

        elif isinstance(device, torch.device):
            self.device = device
            self.args.device = device.type

        else:
            raise ValueError("Invalid device")

    def set_optimizer(self, optimizer) -> None:
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )

        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
            self.args.lr = self.optimizer.defaults["lr"]
            self.args.weight_decay = self.optimizer.defaults.get("weight_decay", 0.0)

        else:
            raise ValueError("Invalid optimizer")

    def set_schedulers(self, schedulers) -> None:
        if schedulers is None:
            self.schedulers = [
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.args.n_epochs
                )
            ]
        else:
            self.schedulers = [schedulers]

    def set_model(self, model: T_model) -> None:
        self.model = model.to(self.device)

    def set_dataset(self, dataset) -> None:
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.is_shuffle,
            num_workers=self.args.num_workers,
        )

    def set_valid_ds(self, valid_ds) -> None:
        if valid_ds is None:
            self.valid_data_loader = None
        else:
            self.valid_data_loader = torch.utils.data.DataLoader(
                valid_ds,
                batch_size=self.args.batch_size,
                shuffle=False,
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

    @abstractmethod
    def step(self, batch) -> Dict:
        # TODO: impl this function
        ...

    def step_info(self, result: Dict) -> None:
        # TODO: impl this function
        ...

    def epoch_info(self) -> None:
        # TODO: impl this function
        ...

    def validate(self) -> None:
        # TODO: impl this function
        ...

    def log2plot(self, key: str) -> None:
        self.logger.plot(key)

    def train(self) -> None:
        # Initialization
        self.model.train()
        threading.Thread(target=self._keyboard_listener, daemon=True).start()

        if self.valid_data_loader is None:
            print(
                colored(
                    "[WARN] No validation dataset provided, skipping validation.",
                    "yellow",
                )
            )

        # Main training loop
        self._stop_training = False
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

                if step_result.get("should_stop"):
                    print(
                        colored(
                            "Training stopped by user command.",
                            "red",
                            attrs=["bold"],
                        )
                    )
                    self._stop_training = True
                    break

            if self._stop_training:
                print(
                    colored(
                        "Training stopped by user command.",
                        "red",
                        attrs=["bold"],
                    )
                )
                break

            for scheduler in self.schedulers:
                scheduler.step()

            self.epoch_info()
            self.logger.save_log(info=False)

            # Validation
            self.model.eval()
            if self.n_epochs % self.args.epochs_per_validation == 0:
                self.validate()
            self.model.train()

            self.n_epochs += 1

        # Finalization
        self.save()
        self.logger.save_log(info=True)
        if self.args.is_draw:
            for key in self.logger.content.__dict__.keys():
                self.logger.plot(key)

    def _keyboard_listener(self) -> None:
        while True:
            user_cmd = input()
            if user_cmd.lower() == "q":
                print(
                    colored(
                        "Keyboard interrupt detected, exiting...",
                        "red",
                        attrs=["bold"],
                    )
                )
                os._exit(0)
            elif user_cmd.lower() == "s":
                print(colored("Saving model...", "light_green"))
                self.save()
