from typing import Any, Callable, Dict, List, Tuple

import torch
from termcolor import colored
from torch import nn

from ...trainer import TrainArgs, Trainer


class OBBDetectionTrainerArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)


class OBBDetectionTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion: Callable,
        args: OBBDetectionTrainerArgs,
        collate_fn: Callable = lambda x: x,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        self.collate_fn: Callable = collate_fn
        super().__init__(
            model, dataset, criterion, args, optimizer, scheduler, valid_ds
        )

    def set_dataset(self, dataset) -> None:
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.is_shuffle,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
        )

    def step(
        self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, Any]:
        image, targets = batch
        image = image.to(self.device)
        self.optimizer.zero_grad()

        pred_cls, pred_reg = self.model(image)
        batch_size, num_queries, num_classes = pred_cls.shape

        bbox_dim = None
        for target in targets:
            if len(target["bboxes"]) > 0:
                bbox_dim = target["bboxes"].shape[-1]
                break

        if bbox_dim is None:
            return {"loss": 0.0}

        all_labels = torch.full(
            (batch_size, num_queries),
            fill_value=0,
            dtype=torch.long,
            device=self.device,
        )
        all_bboxes = torch.zeros(
            (batch_size, num_queries, bbox_dim),
            dtype=torch.float32,
            device=self.device,
        )

        for index, target in enumerate(targets):
            if len(target["bboxes"]) == 0:
                continue
            bboxes = target["bboxes"].to(self.device)
            labels = target["labels"].to(self.device)
            num_objects = min(len(bboxes), num_queries)
            bboxes = bboxes[:num_objects, :]
            labels = labels[:num_objects]
            all_labels[index, :num_objects] = labels
            all_bboxes[index, :num_objects, :] = bboxes

        loss = self.criterion(pred_cls, pred_reg, all_labels, all_bboxes)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def step_info(self, result: Dict[str, Any]) -> None:
        # step
        if self.n_steps % 1000 == 0:
            self.logger.op(
                "step",
                lambda x: {"loss": x.get("loss", 0) + result["loss"]},
                index=self.n_steps,
            )
            print(
                f"(Step {self.n_steps}) "
                + colored("loss", "yellow")
                + f": {self.logger.content.step[f'{self.n_steps}']['loss']}"
            )

        # epoch
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) + result["loss"]},
            index=self.n_epochs,
        )

    def epoch_info(self) -> None:
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) / len(self.data_loader)},
            index=self.n_epochs,
        )
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['loss']}"
        )

    def validate(self) -> None: ...
