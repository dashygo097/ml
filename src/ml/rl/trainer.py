import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

import gymnasium as gym
import torch
from termcolor import colored
from tqdm import tqdm

from ..typical import TrainLogger
from ..utils import load_yaml
from .agents import RLAgent


@dataclass
class RLTrainArgs:
    # Necessary arguments
    device: str  # Device to use for training
    epochs: int  # Number of epochs to train

    optimizer: Dict[str, Any]  # Optimizer options

    # Environment and agent options
    env: Dict[str, Any] = field(default_factory=dict)  # Environment options
    agent: Dict[str, Any] = field(default_factory=dict)  # Agent options

    # Optional arguments with default values
    scheduler: Dict[str, Any] = field(default_factory=dict)  # Scheduler options

    log_dict: str = "./train_logs"  # Directory to save logs
    epochs_per_log: int = 10  # Epochs per log

    save_dict: str = "./checkpoints"  # Directory to save checkpoints

    is_draw: bool = False  # Whether to draw logs
    drawing_list: List[str] = field(default_factory=list)  # List of metrics to draw

    @classmethod
    def from_yaml(cls, path: str) -> "RLTrainArgs":
        data = load_yaml(path)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLTrainArgs":
        return cls(**data)


T_agent = TypeVar("T_agent", bound=RLAgent)
T_env = TypeVar("T_env", bound=gym.Env)
T_args = TypeVar("T_args", bound=RLTrainArgs)


class RLTrainer(Generic[T_agent, T_env, T_args], ABC):
    def __init__(
        self,
        agent: T_agent,
        env: T_env,
        args: T_args,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
    ) -> None:
        self.args = args

        # Setup components
        self._setup_device()
        self._setup_agent(agent)
        self._setup_env(env)
        self._setup_optimizer(optimizer)
        self._setup_scheduler(scheduler)
        self._setup_logging()

        # Training state
        self.n_steps = 0
        self.n_epochs = 0
        self._stop_training = False
        self._terminated = False
        self._truncated = False
        self.best_reward = float("-inf")

    def _setup_device(self) -> None:
        try:
            if self.args.device.lower() == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    self.args.device = "cuda"
                elif torch.mps.is_available():
                    self.device = torch.device("mps")
                    self.args.device = "mps"
                else:
                    self.device = torch.device("cpu")
                    self.args.device = "cpu"
            else:
                self.device = torch.device(self.args.device)

            print(
                colored("│ INFO  │ ", "magenta", attrs=["bold"])
                + colored(f"Using device: {self.device}", "white", attrs=["dark"])
            )
        except RuntimeError as e:
            print(
                colored("│ WARN  │ ", "red", attrs=["bold"])
                + colored(f"Device error: {e}. Falling back to CPU.", "white")
            )
            self.device = torch.device("cpu")
            self.args.device = "cpu"

    def _setup_agent(self, agent: T_agent) -> None:
        self.agent = agent.to(self.device)
        param_count = sum(p.numel() for p in self.agent.parameters())
        trainable_count = sum(
            p.numel() for p in self.agent.parameters() if p.requires_grad
        )
        print(
            colored("│ INFO  │ ", "magenta", attrs=["bold"])
            + colored(
                f"Agent parameters: {param_count:,} ({trainable_count:,} trainable)",
                "white",
                attrs=["dark"],
            )
        )

    def _setup_env(self, env: T_env) -> None:
        self.env = env
        env_name = env.spec.id if env.spec else "Custom"
        print(
            colored("│ INFO  │ ", "magenta", attrs=["bold"])
            + colored(f"Environment: {env_name}", "white", attrs=["dark"])
        )

    def _setup_optimizer(self, optimizer: Optional[type]) -> None:
        if optimizer is None:
            self.optimizer = None
            print(
                colored("│ INFO  │ ", "magenta", attrs=["bold"])
                + colored("No optimizer used", "white", attrs=["dark"])
            )
        else:
            self.optimizer = optimizer(
                self.agent.parameters(),
                **self.args.optimizer,
            )
            print(
                colored("│ OK    │ ", "green", attrs=["bold"])
                + colored(f"Optimizer: {optimizer.__name__}", "white", attrs=["dark"])
            )

    def _setup_scheduler(self, scheduler: Optional[type]) -> None:
        if scheduler is None:
            self.scheduler = None
            print(
                colored("│ INFO  │ ", "magenta", attrs=["bold"])
                + colored("No scheduler used", "white", attrs=["dark"])
            )
        elif isinstance(scheduler, type) and self.optimizer is not None:
            self.scheduler = scheduler(self.optimizer, **self.args.scheduler)
            print(
                colored("│ OK    │ ", "green", attrs=["bold"])
                + colored(f"Scheduler: {scheduler.__name__}", "white", attrs=["dark"])
            )
        else:
            self.scheduler = None

    def _setup_logging(self) -> None:
        os.makedirs(self.args.log_dict, exist_ok=True)
        os.makedirs(self.args.save_dict, exist_ok=True)
        self.logger = TrainLogger(self.args.log_dict)
        print(
            colored("│ INFO  │ ", "magenta", attrs=["bold"])
            + colored(f"Logs directory: {self.args.log_dict}", "white", attrs=["dark"])
        )

    @abstractmethod
    def step(self) -> Dict[str, Any]: ...

    def _process_step_result(self, result: Dict[str, Any]) -> None:
        for key, value in result.items():
            self.logger.op(
                "step",
                lambda x, k=key, v=value: {**x, k: x.get(k, 0) + v},
                index=self.n_steps,
            )

        for key, value in result.items():
            self.logger.op(
                "epoch",
                lambda x, k=key, v=value: {**x, k: x.get(k, 0) + v},
                index=self.n_epochs,
            )

    def _finalize_epoch(self) -> None:
        if str(self.n_epochs) in self.logger.content.epoch:
            epoch_data = self.logger.content.epoch[str(self.n_epochs)]

            if self.n_epochs % self.args.epochs_per_log == 0:
                metrics_parts = []
                for key, value in epoch_data.items():
                    metric_text = colored(f"{key}: ", color="yellow") + colored(
                        f"{value:.4f}", "red", attrs=["dark"]
                    )
                    metrics_parts.append(metric_text)
                metrics_str = "  │ ".join(metrics_parts)
                tqdm.write(
                    colored(f"◆ EPOCH {self.n_epochs:03d}  │ ", "blue", attrs=["bold"])
                    + metrics_str
                )

                # Track best reward
                if "reward" in epoch_data and epoch_data["reward"] > self.best_reward:
                    best_marker = " " + colored("★ BEST ★", "red", attrs=["bold"])
                    tqdm.write(colored("   ", "blue") + best_marker)
                    self.best_reward = epoch_data["reward"]

    def save_checkpoint(self, tag: str = "") -> str:
        os.makedirs(self.args.save_dict, exist_ok=True)

        if tag:
            filename = f"checkpoint_{tag}.pt"
        else:
            filename = f"checkpoint_epoch{self.n_epochs:03d}_step{self.n_steps:06d}.pt"

        path = os.path.join(self.args.save_dict, filename)

        checkpoint = {
            "epoch": self.n_epochs,
            "step": self.n_steps,
            "agent_state": self.agent.state_dict(),
            "args": self.args,
        }

        if self.optimizer is not None:
            checkpoint["optimizer_state"] = self.optimizer.state_dict()

        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print()
        print(
            colored("│ OK    │ ", "green", attrs=["bold"])
            + colored(f"Checkpoint saved: {path}", "white", attrs=["bold"])
        )

        return path

    def _save_checkpoint(self, best: bool = False) -> None:
        tag = "best" if best else f"epoch{self.n_epochs:03d}"
        self.save_checkpoint(tag=tag)

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.agent.load_state_dict(checkpoint["agent_state"])
        self.n_epochs = checkpoint.get("epoch", 0)
        self.n_steps = checkpoint.get("step", 0)

        if load_optimizer and self.optimizer is not None:
            if "optimizer_state" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if self.scheduler is not None and "scheduler_state" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(
            colored("│ OK    │ ", "green", attrs=["bold"])
            + colored(f"Checkpoint loaded: {path}", "white", attrs=["bold"])
        )

    def train(self, resume_from: Optional[str] = None) -> None:
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        listener_thread = threading.Thread(
            target=self._keyboard_listener,
            daemon=True,
        )
        listener_thread.start()

        if self.optimizer is None:
            print(
                colored("│ WARN  │ ", "red", attrs=["bold"])
                + colored("No optimizer provided", "white", attrs=["bold"])
            )

        print()

        try:
            self.agent.train()

            for epoch in range(self.n_epochs, self.args.epochs):
                self.n_epochs = epoch

                if self._stop_training:
                    print(
                        colored("│ INFO  │ ", "magenta", attrs=["bold"])
                        + colored("Training stopped by user", "white", attrs=["bold"])
                    )
                    break

                self._train_episode()

                if self.scheduler is not None:
                    self.scheduler.step()

                self._finalize_epoch()

                if self.n_epochs % self.args.epochs_per_log == 0:
                    self.logger.save_log(info=False)

        except KeyboardInterrupt:
            print(
                colored("│ WARN  │ ", "red", attrs=["bold"])
                + colored("Training interrupted by user", "white", attrs=["bold"])
            )

        finally:
            # Finalization
            self.agent.eval()
            self.env.close()
            self.save_checkpoint(tag="final")
            self.logger.save_log(info=True)

            if self.args.is_draw:
                for key in self.logger.content.__dict__.keys():
                    try:
                        self.logger.plot(key)
                    except Exception as e:
                        print(
                            colored("│ WARN  │ ", "red", attrs=["bold"])
                            + colored(f"Failed to plot {key}: {e}", "white")
                        )

            print()
            print(
                colored("  ✓ TRAINING COMPLETED SUCCESSFULLY", "green", attrs=["bold"])
            )
            print()

    def _train_episode(self) -> None:
        self.agent.train()

        pbar = tqdm(
            total=1,
            desc=colored(
                f"  ⟳ Epoch {self.n_epochs:03d}/{self.args.epochs:03d}",
                "red",
                attrs=["bold"],
            ),
            position=0,
            leave=False,
        )

        try:
            # Reset environment and agent
            self._terminated = False
            self._truncated = False
            self._obs, self._info = self.env.reset(options=self.args.env)
            self.agent.reset(options=self.args.agent)

            # Run episode
            step_count = 0
            while not (self._terminated or self._truncated):
                try:
                    step_result = self.step()
                    self._process_step_result(step_result)
                    self.n_steps += 1
                    step_count += 1

                    if self._stop_training:
                        break

                except Exception as e:
                    print(
                        colored("│ ERROR │ ", "red", attrs=["dark"])
                        + colored(f"Step failed: {e}", "white")
                    )
                    raise

            pbar.update(1)

        finally:
            pbar.close()

    def _keyboard_listener(self) -> None:
        print(
            colored("│ INFO  │ ", "magenta", attrs=["dark"])
            + colored("Press 's' to save, 'q' to quit", "white", attrs=["bold"])
        )

        while True:
            try:
                cmd = input().strip().lower()

                if cmd == "q":
                    print(
                        colored("│ INFO  │ ", "magenta", attrs=["bold"])
                        + colored("Quit signal received", "white", attrs=["bold"])
                    )
                    self._stop_training = True
                    break

                elif cmd == "s":
                    print(
                        colored("│ INFO  │ ", "magenta", attrs=["bold"])
                        + colored("Save signal received", "white", attrs=["bold"])
                    )
                    self.save_checkpoint()

                elif cmd:
                    print(
                        colored("│ WARN  │ ", "red", attrs=["bold"])
                        + colored(
                            f"Unknown command: {cmd} (valid: 's' or 'q')", "white"
                        )
                    )

            except EOFError:
                break
            except Exception as e:
                print(
                    colored("│ WARN  │ ", "red", attrs=["bold"])
                    + colored(f"Keyboard listener error: {e}", "white")
                )
