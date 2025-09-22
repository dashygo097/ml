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

    def set_valid_ds(self, valid_ds) -> None:
        if valid_ds is None:
            self.valid_data_loader = None
        else:
            self.valid_data_loader = torch.utils.data.DataLoader(
                valid_ds,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=self.collate_fn,
            )

    def step(
        self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, Any]:
        image, targets = batch
        image = image.to(self.device)
        self.optimizer.zero_grad()

        pred_cls, pred_bbox, pred_angle = self.model(image)
        batch_size, num_queries, num_classes = pred_cls.shape

        bbox_dim = None
        for target in targets:
            if len(target["bboxes"]) > 0:
                bbox_dim = target["bboxes"].shape[-1] - 1
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
        all_angles = torch.zeros(
            (batch_size, num_queries, 1),
            dtype=torch.float32,
            device=self.device,
        )

        for index, target in enumerate(targets):
            if len(target["bboxes"]) == 0:
                continue
            bboxes = target["bboxes"].to(self.device)
            labels = target["labels"].to(self.device)
            num_objects = min(len(bboxes), num_queries)
            bboxes = bboxes[:num_objects, :bbox_dim]
            labels = labels[:num_objects]
            all_labels[index, :num_objects] = labels
            all_bboxes[index, :num_objects, :bbox_dim] = bboxes

            angles = target["bboxes"][:num_objects, bbox_dim:].to(self.device)
            all_angles[index, :num_objects] = angles

        loss, cls_loss, bbox_loss, angle_loss = self.criterion(
            pred_cls, pred_bbox, pred_angle, all_labels, all_bboxes, all_angles
        )
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "bbox_loss": bbox_loss.item(),
            "angle_loss": angle_loss.item(),
        }

    def step_info(self, result: Dict[str, Any]) -> None:
        # step
        if self.n_steps % 10 == 0 and self.n_steps > 0:
            self.logger.op(
                "step",
                lambda x: {
                    "loss": x.get("loss", 0) + result["loss"],
                    "cls_loss": x.get("cls_loss", 0) + result["cls_loss"],
                    "bbox_loss": x.get("bbox_loss", 0) + result["bbox_loss"],
                    "angle_loss": x.get("angle_loss", 0) + result["angle_loss"],
                },
                index=self.n_steps,
            )
            print(
                f"(Step {self.n_steps}) "
                + colored("loss", "yellow")
                + f": {self.logger.content.step[f'{self.n_steps}']['loss']}, "
                + colored("cls_loss", "cyan")
                + f": {self.logger.content.step[f'{self.n_steps}']['cls_loss']}, "
                + colored("bbox_loss", "magenta")
                + f": {self.logger.content.step[f'{self.n_steps}']['bbox_loss']}, "
                + colored("angle_loss", "green")
                + f": {self.logger.content.step[f'{self.n_steps}']['angle_loss']}"
            )

        # epoch
        self.logger.op(
            "epoch",
            lambda x: {
                "loss": x.get("loss", 0) + result["loss"],
                "cls_loss": x.get("cls_loss", 0) + result["cls_loss"],
                "bbox_loss": x.get("bbox_loss", 0) + result["bbox_loss"],
                "angle_loss": x.get("angle_loss", 0) + result["angle_loss"],
            },
            index=self.n_epochs,
        )

    def epoch_info(self) -> None:
        self.logger.op(
            "epoch",
            lambda x: {
                "loss": x.get("loss", 0) / len(self.data_loader),
                "cls_loss": x.get("cls_loss", 0) / len(self.data_loader),
                "bbox_loss": x.get("bbox_loss", 0) / len(self.data_loader),
                "angle_loss": x.get("angle_loss", 0) / len(self.data_loader),
            },
            index=self.n_epochs,
        )
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['loss']}, "
            + colored("cls_loss", "cyan")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['cls_loss']}, "
            + colored("bbox_loss", "magenta")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['bbox_loss']}, "
            + colored("angle_loss", "green")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['angle_loss']}"
        )

    def validate(self) -> None: ...
