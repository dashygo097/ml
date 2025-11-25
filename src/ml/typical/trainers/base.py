import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import torch
from termcolor import colored
from torch import nn
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from ...utils import load_yaml
from .logger import TrainLogger


@dataclass
class TrainArgs:
    # Necessary arguments
    device: str  # Device to use for training

    batch_size: int  # Batch size for training
    epochs: int  # Number of epochs to train

    optimizer: Dict[str, Any]  # Optimizer options

    # Optional arguments with default values
    num_workers: int = 0  # Number of workers for data loading
    is_shuffle: bool = True  # Whether to shuffle the dataset

    epochs_per_validation: int = 1  # Epochs per validation
    unfreeze_epoch: int = -1  # Epoch to unfreeze the model, -1 means never

    log_dict: str = "./train_logs"  # Directory to save logs
    steps_per_log: int = 1_000_000_000  # Steps per log(Default: no log)
    epochs_per_log: int = 1  # Epochs per log

    save_dict: str = "./checkpoints"  # Directory to save checkpoints

    scheduler: Dict[str, Any] = field(default_factory=dict)  # Scheduler options

    is_draw: bool = False  # Whether to draw logs
    drawing_list: List[str] = field(default_factory=list)  # List of metrics to draw

    @classmethod
    def from_yaml(cls, path: str) -> "TrainArgs":
        data = load_yaml(path)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainArgs":
        return cls(**data)


T_model = TypeVar("T_model", bound=nn.Module)
T_args = TypeVar("T_args", bound=TrainArgs)


class Trainer(ABC, Generic[T_model, T_args]):
    def __init__(
        self,
        model: T_model,
        train_ds: Any,
        loss_fn: Callable,
        args: T_args,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.args = args

        # Setup components
        self._setup_device()
        self._setup_model()
        self._setup_dataloaders(train_ds, valid_ds)
        self._setup_optimizer(optimizer)
        self._setup_scheduler(scheduler)
        self._setup_logging()
        self._setup_grad_scaler()

        # Training state
        self.n_steps = 0
        self.n_epochs = 0
        self._stop_training = False
        self.best_val_loss = float("inf")

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

    def _setup_model(self) -> None:
        self.model = self.model.to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            colored("│ INFO  │ ", "magenta", attrs=["bold"])
            + colored(
                f"Model parameters: {param_count:,} ({trainable_count:,} trainable)",
                "white",
                attrs=["dark"],
            )
        )

    def _setup_dataloaders(self, train_ds: Any, valid_ds: Optional[Any]) -> None:
        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=self.args.is_shuffle,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        if valid_ds is not None:
            self.val_loader = torch.utils.data.DataLoader(
                valid_ds,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=self.device.type == "cuda",
            )
        else:
            self.val_loader = None

        print(
            colored("│ INFO  │ ", "magenta", attrs=["bold"])
            + colored(
                f"Train samples: {len(train_ds):,}  │ Valid samples: {len(valid_ds) if valid_ds else 'N/A'}",
                "white",
                attrs=["dark"],
            )
        )

    def _setup_optimizer(self, optimizer: Optional[type]) -> None:
        if optimizer is None:
            optimizer = torch.optim.Adam

        self.optimizer = optimizer(
            self.model.parameters(),
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
        else:
            self.scheduler = scheduler(self.optimizer, **self.args.scheduler)
            print(
                colored("│ OK    │ ", "green", attrs=["bold"])
                + colored(f"Scheduler: {scheduler.__name__}", "white", attrs=["dark"])
            )

    def _setup_logging(self) -> None:
        os.makedirs(self.args.log_dict, exist_ok=True)
        os.makedirs(self.args.save_dict, exist_ok=True)
        self.logger = TrainLogger(self.args.log_dict)

    def _setup_grad_scaler(self) -> None:
        try:
            self.scaler = GradScaler(self.args.device)
            print(
                colored("│ OK    │ ", "green", attrs=["bold"])
                + colored("Gradient scaler enabled", "white", attrs=["dark"])
            )
        except Exception as e:
            self.scaler = None
            print(
                colored("│ WARN  │ ", "red", attrs=["bold"])
                + colored(f"Gradient scaler disabled: {e}", "white")
            )

    @abstractmethod
    def step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]
    ) -> Dict[str, float]: ...

    @abstractmethod
    def validate_step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]
    ) -> Dict[str, float]: ...

    def _process_step_result(self, result: Dict[str, float]) -> None:
        for key, value in result.items():
            self.logger.op(
                "step",
                lambda x, k=key, v=value: {**x, k: x.get(k, 0) + v},
                index=self.n_steps,
            )

        if self.n_steps % self.args.steps_per_log == 0 and self.n_steps > 0:
            step_data = self.logger.content.step[f"{self.n_steps}"]
            metrics_parts = []
            for key, value in step_data.items():
                metric_text = f"{key}: " + colored(
                    f"{value:.4f}", "yellow", attrs=["dark"]
                )
                metrics_parts.append(metric_text)
            metrics_str = "  │ ".join(metrics_parts)
            tqdm.write(
                colored(f"✦ Step {self.n_steps:06d}  │ ", "blue", attrs=["bold"])
                + metrics_str
            )

        for key, value in result.items():
            self.logger.op(
                "epoch",
                lambda x, k=key, v=value: {**x, k: x.get(k, 0) + v},
                index=self.n_epochs,
            )

    def validate(self) -> None:
        if self.val_loader is None:
            return

        self.model.eval()
        val_metrics = {}

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=colored("  ⟳ Validating", "magenta", attrs=["bold"]),
                leave=False,
            )

            for batch in pbar:
                batch_metrics = self.validate_step(batch)

                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)

        avg_metrics = {k: sum(v) / len(v) for k, v in val_metrics.items()}

        self.logger.log("valid", avg_metrics, self.n_epochs)

        metrics_parts = []
        for key, value in avg_metrics.items():
            metric_text = colored(f"{key}: ", color="yellow") + colored(
                f"{value:.4f}", "green", attrs=["dark"]
            )
            metrics_parts.append(metric_text)
        metrics_str = "  │ ".join(metrics_parts)

        best_marker = ""
        if "loss" in avg_metrics and avg_metrics["loss"] < self.best_val_loss:
            best_marker = " " + colored("★ BEST ★", "red", attrs=["bold"])

        tqdm.write(
            colored("✓ VALIDATION  │ ", "green", attrs=["bold"])
            + metrics_str
            + best_marker
        )

        if "loss" in avg_metrics:
            if avg_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = avg_metrics["loss"]
                self._save_checkpoint(best=True)

        self.model.train()

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
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "args": self.args,
        }

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

        self.model.load_state_dict(checkpoint["model_state"])
        self.n_epochs = checkpoint.get("epoch", 0)
        self.n_steps = checkpoint.get("step", 0)

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if self.scheduler is not None and "scheduler_state" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(
            colored("│ OK    │ ", "green", attrs=["bold"])
            + colored(f"Checkpoint loaded: {path}", "white", attrs=["bold"])
        )

    def unfreeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
        print(
            colored("│ WARN  │ ", "red", attrs=["bold"])
            + colored(
                "Model unfrozen - all parameters trainable", "white", attrs=["bold"]
            )
        )

        trainable_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_count = sum(p.numel() for p in self.model.parameters())
        print(
            colored("│ INFO  │ ", "magenta", attrs=["bold"])
            + colored(
                f"Trainable parameters: {trainable_count:,} / {total_count:,}",
                "white",
                attrs=["dark"],
            )
        )

        self.optimizer = type(self.optimizer)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.args.optimizer,
        )
        print(
            colored("│ OK    │ ", "green", attrs=["bold"])
            + colored(
                "Optimizer updated with unfrozen parameters", "white", attrs=["dark"]
            )
        )

    def train(self, resume_from: Optional[str] = None) -> None:
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        listener_thread = threading.Thread(
            target=self._keyboard_listener,
            daemon=True,
        )
        listener_thread.start()

        if self.val_loader is None:
            print(
                colored("│ WARN  │ ", "red", attrs=["bold"])
                + colored("No validation dataset provided", "white", attrs=["bold"])
            )

        print()

        try:
            self.model.train()

            for epoch in range(self.n_epochs, self.args.epochs):
                self.n_epochs = epoch

                if self.args.unfreeze_epoch >= 0 and epoch == self.args.unfreeze_epoch:
                    self.unfreeze_backbone()

                if self._stop_training:
                    print(
                        colored("│ INFO  │ ", "magenta", attrs=["bold"])
                        + colored("Training stopped by user", "white", attrs=["bold"])
                    )
                    break

                self._train_epoch()

                if self.scheduler is not None:
                    self.scheduler.step()

                if epoch % self.args.epochs_per_validation == 0:
                    self.validate()

                self.logger.save_log()

        except KeyboardInterrupt:
            print(
                colored("│ WARN  │ ", "red", attrs=["bold"])
                + colored("Training interrupted by user", "white", attrs=["bold"])
            )

        finally:
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

    def _train_epoch(self) -> None:
        self.model.train()

        pbar = tqdm(
            self.train_loader,
            desc=colored(
                f"  ⟳ Epoch {self.n_epochs:03d}/{self.args.epochs:03d}",
                "red",
                attrs=["bold"],
            ),
            position=0,
            leave=False,
        )

        try:
            for batch in pbar:
                try:
                    step_result = self.step(batch)
                    self._process_step_result(step_result)
                    self.n_steps += 1

                    if self._stop_training:
                        break

                except Exception as e:
                    print(
                        colored("│ ERROR  │ ", "red", attrs=["dark"])
                        + colored(f"Step failed: {e}", "white")
                    )
                    raise

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
                        colored(f"│ WARN  │ ", "red", attrs=["bold"])
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
