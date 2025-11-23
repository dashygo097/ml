from typing import Any, Callable, Dict, Optional

import torch.nn as nn
from torch.cuda.amp import autocast

from ....trainer import TrainArgs, Trainer


class GPTrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)
        self.use_fp16: bool = self.args.get("fp16", False)


class GPTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        dataset,
        loss_fn: Callable,
        args: GPTrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
        valid_ds: Optional[Any] = None,
    ) -> None:
        super().__init__(model, dataset, loss_fn, args, optimizer, scheduler, valid_ds)
        self.set_tokenizer(tokenizer)

    def set_tokenizer(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        # TODO: Implement dataset handling

    def step(self, batch: Dict) -> Dict[str, Any]:
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

            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            loss.backward()
            self.optimizer.step()

        else:
            with autocast():
                logits = self.model(inputs_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs_ids[..., 1:].contiguous()
                loss = self.loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                loss.backward()
                self.optimizer.step()

        return {"loss": loss.item()}

    def validate(self) -> None: ...
