import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from termcolor import colored

from ...base import BaseDataset


class DepthEstDatasetL0(BaseDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable = T.ToTensor(),
        label_transform: Callable = T.ToTensor(),
    ) -> None:
        super().__init__(root, split, transform)
        self.label_transform = label_transform
        self.image_dir = os.path.join(root, "images", split)
        self.label_dir = os.path.join(root, "labels", split)

        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise ValueError(f"Annotation directory not found: {self.label_dir}")

        self.image_paths = sorted(
            [
                os.path.join(self.image_dir, fname)
                for fname in os.listdir(self.image_dir)
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

        assert len(self.image_paths) == len(self.label_paths), colored(
            "[ERROR] Mismatch between the number of images from images and labels: "
            + f"{len(self.image_paths)} vs {len(self.label_paths)}",
            color="red",
            attrs=["bold"],
        )

    def __len__(self) -> int:
        return len(self.label_paths)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]:
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        image = self.transform(image)
        label = self.label_transform(label)

        return (image, label), {}
