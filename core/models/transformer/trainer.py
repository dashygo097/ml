from typing import Dict

import torch.nn as nn

from ...trainer import TrainArgs, Trainer


class GPTrainArgs(TrainArgs):
    def __init__(self, path: str) -> None:
        super().__init__(path)


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

        logits = self.model(inputs_ids)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs_ids[..., 1:].contiguous()

        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return {"loss": loss}

    def step_info(self, result: Dict) -> None: ...

    def epoch_info(self) -> None: ...

    def validate(self) -> None: ...
