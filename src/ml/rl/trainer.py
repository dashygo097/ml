import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

import gymnasium as gym
import torch
from termcolor import colored
from tqdm import tqdm

from ..logger import TrainLogger
from ..utils import load_yaml
from .agents import RLAgent


class RLTrainArgs:
    def __init__(self, path_or_dict: str | Dict) -> None:
        self.args: Dict = (
            load_yaml(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
        )
        self.device: str = self.args["device"]
        self.epochs: int = self.args["epochs"]

        # agent/env reset options
        self.env_options: Dict = self.args.get("env", {})
        self.agent_options: Dict = self.args.get("agent", {})

        # optimizer/scheduler options
        self.optimizer_options: Dict = self.args.get("optimizer", {})
        self.scheduler_options: Dict = self.args.get("scheduler", {})

        # general training options
        self.lr: float = (
            self.optimizer_options.get("lr", 1e-3)
            if hasattr(self.args, "optimizer")
            else self.args.get("lr", 1e-3)
        )
        self.weight_decay: float = (
            self.optimizer_options.get("weight_decay", 0.0)
            if hasattr(self.args, "optimizer")
            else self.args.get("weight_decay", 0.0)
        )

        self.save_dict: str = self.args.get("save_dict", "./checkpoints")

        if "log_dict" not in self.args["info"].keys():
            self.log_dict: str = "./train_logs"
            if self.log_dict.endswith("/"):
                self.log_dict = self.log_dict[:-1]

        else:
            self.log_dict: str = self.args["info"]["log_dict"]

        self.drawing_list: List[str] = self.args["info"].get("drawing_list", [])
        self.is_draw: bool = self.args["info"].get("is_draw", False)


T_env = TypeVar("T_env", bound=gym.Env)
T_agent = TypeVar("T_agent", bound=RLAgent)


class RLTrainer(Generic[T_env, T_agent], ABC):
    def __init__(
        self,
        agent: T_agent,
        env: T_env,
        args: RLTrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
    ) -> None:
        self.args: RLTrainArgs = args

        self.set_device(args.device)
        self.set_agent(agent)
        self.set_env(env)
        self.set_optimizer(optimizer)
        self.set_scheduler(scheduler)

        self.n_steps: int = 0
        self.n_epochs: int = 0
        self.logger: TrainLogger = TrainLogger(self.args.log_dict)

        self._terminated: bool = False
        self._truncated: bool = False

    def set_device(self, device: str) -> None:
        self.device = device

    def set_agent(self, agent: T_agent) -> None:
        self.agent = agent.to(self.device)

    def set_env(self, env: T_env) -> None:
        self.env = env

    def set_optimizer(self, optimizer: Optional[type]) -> None:
        if optimizer is None:
            self.optimizer = None
        else:
            self.optimizer = optimizer(
                self.agent.parameters(),
                **self.args.optimizer_options,
            )

    def set_scheduler(self, scheduler: Optional[type]) -> None:
        if scheduler is None:
            self.schedulers = None
        elif isinstance(scheduler, type) and self.optimizer is not None:
            self.schedulers = [scheduler(self.optimizer, **self.args.scheduler_options)]

    def save(self) -> None:
        os.makedirs(self.args.save_dict, exist_ok=True)
        path = os.path.join(self.args.save_dict, f"checkpoint_{self.n_epochs}.pt")
        torch.save(self.agent.state_dict(), path)
        print(f"[INFO] agent saved at: {path}!")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file {path} does not exist.")
        self.agent.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[INFO] agent loaded from: {path}!")

    @abstractmethod
    def step(self) -> Dict[str, Any]: ...

    def step_info(self, result: Dict) -> None:
        # TODO: impl this function
        ...

    def epoch_info(self) -> None:
        # TODO: impl this function
        ...

    def should_stop(self) -> None:
        self._stop_training = True

    def log2plot(self, key: str) -> None:
        self.logger.plot(key)

    def train(self) -> None:
        # Initialization
        self.agent.train()
        threading.Thread(target=self._keyboard_listener, daemon=True).start()

        # Main training loop
        self._stop_training = False
        for _ in tqdm(
            range(self.args.epochs),
            desc=colored("Training", "light_red", attrs=["bold"]),
            leave=False,
        ):
            self._terminated = False
            self._truncated = False
            self._obs, self._info = self.env.reset(options=self.args.env_options)
            self.agent.reset(options=self.args.agent_options)
            while not (self._terminated or self._truncated):
                step_result = self.step()
                self.step_info(step_result)
                self.n_steps += 1

            if self._stop_training:
                print(
                    colored(
                        "Training stopped by user command.",
                        "red",
                        attrs=["bold"],
                    )
                )
                break

            if self.schedulers is not None:
                for scheduler in self.schedulers:
                    scheduler.step()

            self.epoch_info()
            self.logger.save_log()

            self.n_epochs += 1

        # Finalization
        self.agent.eval()
        self.env.close()
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
                print(colored("Saving agent...", "light_green"))
                self.save()
