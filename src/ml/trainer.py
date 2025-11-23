import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import torch
from termcolor import colored
from torch import nn
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from .logger import TrainLogger
from .utils import load_yaml


class TrainArgs:
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        self.args: Dict[str, Any] = (
            load_yaml(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
        )
        self.device: str = self.args["device"]
        self.batch_size: int = self.args["batch_size"]
        self.epochs: int = self.args["epochs"]
        self.num_workers: int = self.args.get("num_workers", 0)

        # optimizer/scheduler options
        assert "optimizer" in self.args.keys(), "`optimizer` must be specified in args!"
        self.optimizer_options: Dict[str, Any] = self.args.get("optimizer", {})
        self.scheduler_options: Dict[str, Any] = self.args.get("scheduler", {})

        assert "lr" in self.optimizer_options.keys(), (
            "`lr` must be specified in optimizer options!"
        )
        self.lr: float = self.optimizer_options.get("lr", 0.0)
        self.weight_decay: float = self.optimizer_options.get("weight_decay", 0.0)

        self.is_shuffle: bool = self.args.get("is_shuffle", False)
        self.save_dict: str = self.args.get("save_dict", "./checkpoints")

        self.epochs_per_validation: int = self.args.get("epochs_per_validation", 1)
        self.unfreeze_epoch: int = self.args.get("unfreeze_epoch", -1)

        if "log_dict" not in self.args["info"].keys():
            self.log_dict: str = "./train_logs"
            if self.log_dict.endswith("/"):
                self.log_dict = self.log_dict[:-1]

        else:
            self.log_dict: str = self.args["info"]["log_dict"]

        self.steps_per_log: int = self.args.get("steps_per_log", 10)
        self.epochs_per_log: int = self.args.get("epochs_per_log", 1)
        self.drawing_list: List[str] = self.args["info"].get("drawing_list", [])
        self.is_draw: bool = self.args["info"].get("is_draw", False)


T_model = TypeVar("T_model", bound=nn.Module)
T_args = TypeVar("T_args", bound=TrainArgs)


class Trainer(ABC, Generic[T_model, T_args]):
    def __init__(
        self,
        model: T_model,
        dataset: Any,
        loss_fn: Callable,
        args: T_args,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        self.model: T_model = model
        self.loss_fn: Callable = loss_fn
        self.args: T_args = args

        self.set_device(args.device)
        self.set_model(model)
        self.set_dataset(dataset)
        self.set_optimizer(optimizer)
        self.set_scheduler(scheduler)
        self.set_valid_ds(valid_ds)
        try:
            self.scaler = GradScaler(self.args.device)
        except Exception:
            print(
                colored(
                    f"[WARN] `{self.args.device}` is NOT available, scalar won't be used.",
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

    def set_optimizer(self, optimizer: Optional[type]) -> None:
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), **self.args.optimizer_options
            )
        else:
            self.optimizer = optimizer(
                self.model.parameters(),
                **self.args.optimizer_options,
            )

    def set_scheduler(self, scheduler: Optional[type]) -> None:
        if scheduler is None:
            self.scheduler = None
        elif isinstance(scheduler, type) and self.optimizer is not None:
            self.scheduler = scheduler(self.optimizer, **self.args.scheduler_options)

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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file {path} does not exist.")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[INFO] Model loaded from: {path}!")

    @abstractmethod
    def step(self, batch: Any) -> Dict[str, Any]:
        # TODO: impl this function
        ...

    def step_info(self, result: Dict[str, Any]) -> None:
        if self.n_steps % self.args.steps_per_log == 0 and self.n_steps > 0:
            for key, value in result.items():
                self.logger.op(
                    "step",
                    lambda x, k=key, v=value: {**x, k: x.get(k, 0) + v},
                    index=self.n_steps,
                )
            
            step_data = self.logger.content.step[f'{self.n_steps}']
            metrics_str = ", ".join([
                f"{colored(key, 'yellow')}: {value:.4f}" 
                for key, value in step_data.items()
            ])
            tqdm.write(f"(Step {self.n_steps}) {metrics_str}")

        for key, value in result.items():
            self.logger.op(
                "epoch",
                lambda x, k=key, v=value: {**x, k: x.get(k, 0) + v},
                index=self.n_epochs,
            )

    def epoch_info(self) -> None:
        for key in self.logger.content.epoch[f'{self.n_epochs}'].keys():
            self.logger.op(
                "epoch",
                lambda x, k=key: {**x, k: x.get(k, 0) / len(self.data_loader)},
                index=self.n_epochs,
            )
        
        epoch_data = self.logger.content.epoch[f'{self.n_epochs}']
        metrics_str = ", ".join([
            f"{colored(key, 'yellow')}: {value:.4f}" 
            for key, value in epoch_data.items()
        ])
        tqdm.write(f"(Epoch {self.n_epochs}) {metrics_str}")

    def validate(self) -> None:
        # TODO: impl this function
        ...

    def should_stop(self) -> None:
        self._stop_training = True

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
        for epoch in range(self.args.epochs):
            # Unfreeze model if needed
            if self.args.unfreeze_epoch >= 0 and epoch == self.args.unfreeze_epoch:
                for param in self.model.parameters():
                    param.requires_grad = True
                print(colored("Model unfrozen.", "light_green"))

            # Check for stop command
            if self._stop_training:
                tqdm.write(
                    colored(
                        "Training stopped by user command.",
                        "red",
                        attrs=["bold"],
                    )
                )
                break

            pbar = tqdm(
                self.data_loader,
                total=len(self.data_loader),
                desc=colored(f"epoch: {epoch}", "light_red", attrs=["bold"]),
                position=0,
                leave=True,
            )

            # Training
            for batch in pbar:
                step_result = self.step(batch)
                self.step_info(step_result)
                self.n_steps += 1

                if self._stop_training:
                    break

            pbar.close()

            if self.scheduler is not None:
                self.scheduler.step()

            self.epoch_info()
            self.logger.save_log()

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
                tqdm.write(
                    colored(
                        "Keyboard interrupt detected, exiting...",
                        "red",
                        attrs=["bold"],
                    )
                )
                os._exit(0)
            elif user_cmd.lower() == "s":
                tqdm.write(colored("Saving model...", "light_green"))
                self.save()
