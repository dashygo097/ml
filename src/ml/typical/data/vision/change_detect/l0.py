import os
from typing import Any, Callable, Dict, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from termcolor import colored

from ...base import BaseDataset


class ChangeDetectionDatasetL0(BaseDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable = T.ToTensor(),
        label_transform: Callable = T.ToTensor(),
    ):
        super().__init__(root, split, transform)
        self.label_transform = label_transform
        self.image_time1_dir = os.path.join(root, "images", split, "time1")
        self.image_time2_dir = os.path.join(root, "images", split, "time2")
        self.label_dir = os.path.join(root, "labels", split)

        if not os.path.exists(self.image_time1_dir):
            raise ValueError(f"Image directory not found: {self.image_time1_dir}")
        if not os.path.exists(self.image_time2_dir):
            raise ValueError(f"Image directory not found: {self.image_time2_dir}")
        if not os.path.exists(self.label_dir):
            raise ValueError(f"Annotation directory not found: {self.label_dir}")

        self.image_time1_paths = sorted(
            [
                os.path.join(self.image_time1_dir, fname)
                for fname in os.listdir(self.image_time1_dir)
                if fname.endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.image_time2_paths = sorted(
            [
                os.path.join(self.image_time2_dir, fname)
                for fname in os.listdir(self.image_time2_dir)
                if fname.endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.label_paths = sorted(
            [
                os.path.join(self.label_dir, fname)
                for fname in os.listdir(self.label_dir)
                if fname.endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        assert len(self.image_time1_paths) == len(self.image_time2_paths), colored(
            "[ERROR] Mismatch between the number of images from time1 and time2: "
            + f"{len(self.image_time1_paths)} vs {len(self.image_time2_paths)}",
            color="red",
            attrs=["bold"],
        )
        assert len(self.image_time1_paths) == len(self.label_paths), colored(
            "[ERROR] Mismatch between the number of images from images and labels: "
            + f"{len(self.image_time1_paths)} vs {len(self.label_paths)}",
            color="red",
            attrs=["bold"],
        )

    def __len__(self) -> int:
        return len(self.label_paths)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]:
        img_time1_path = self.image_time1_paths[idx]
        img_time2_path = self.image_time2_paths[idx]
        label_path = self.label_paths[idx]
        image_time1 = Image.open(img_time1_path).convert("RGB")
        image_time2 = Image.open(img_time2_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        image_time1 = self.transform(image_time1)
        image_time2 = self.transform(image_time2)
        label = self.label_transform(label)

        return (image_time1, image_time2, label), {}
