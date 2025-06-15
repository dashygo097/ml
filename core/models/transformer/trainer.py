from typing import Dict

import torch.nn as nn
from termcolor import colored
from torch.cuda.amp import autocast

from ...trainer import TrainArgs, Trainer


class GPTrainArgs(TrainArgs):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.use_fp16 = self.args.get("fp16", False)


class GPTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        dataset,
        criterion,
        args: GPTrainArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )
        self.set_tokenizer(tokenizer)

    def set_tokenizer(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    # TODO: Implement dataset handling

    def step(self, batch: Dict) -> Dict:
        # TODO: Implement step handling
        if len(batch) == 2:
            inputs_ids, mask = batch
            labels = None

            inputs_ids, mask = inputs_ids.to(self.device), mask.to(self.device)

        elif len(batch) == 3:
            inputs_ids, mask, labels = batch

            inputs_ids, mask, labels = (
                inputs_ids.to(self.device),
                mask.to(self.device),
                labels.to(self.device),
            )

        else:
            raise ValueError("Batch must contain 2 or 3 elements.")

        self.optimizer.zero_grad()

        if not self.args.use_fp16:
            logits = self.model(inputs_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs_ids[..., 1:].contiguous()

            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            loss.backward()
            self.optimizer.step()

        else:
            with autocast():
                logits = self.model(inputs_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs_ids[..., 1:].contiguous()
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return {"loss": loss.item()}

    def step_info(self, result: Dict) -> None:
        # step
        step_info = self.logger["step"]
        step_checkpoint = self.n_steps // 1000
        if f"step {self.n_steps}" not in step_info and self.n_steps % 1000 == 0:
            step_info[f"step {self.n_steps}"] = {}
            step_info[f"step {self.n_steps}"]["loss"] = 0.0
        step_info[f"step {step_checkpoint * 1000}"]["loss"] += (
            float(result["loss"]) / 1000
        )
        self.logger["step"] = step_info

        if self.n_steps % 1000 == 0:
            print(
                f"(Step {self.n_steps}) "
                + colored("loss", "yellow")
                + f": {self.logger['step'][f'step {self.n_steps}']['loss']}"
            )
            self.save_log(info=False)

        # epoch
        epoch_logger = self.logger["epoch"]
        if f"epoch {self.n_epochs}" not in epoch_logger:
            epoch_logger[f"epoch {self.n_epochs}"] = {}
            epoch_logger[f"epoch {self.n_epochs}"]["loss"] = 0.0

        epoch_logger[f"epoch {self.n_epochs}"]["loss"] += float(result["loss"])

    def epoch_info(self) -> None:
        self.logger["epoch"][f"epoch {self.n_epochs}"]["loss"] /= len(self.data_loader)
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {self.logger['epoch'][f'epoch {self.n_epochs}']['loss']}"
        )

        if self.n_epochs % 20 == 0 and self.n_epochs > 0:
            self.save()

        self.save_log(info=False)

    def validate(self) -> None: ...
