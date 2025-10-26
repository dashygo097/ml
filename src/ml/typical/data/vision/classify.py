import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from termcolor import colored

from ....dataset import BaseDataset


class ImageClassificationDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
    ):
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
                if fname.endswith(".txt")
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
        label = open(label_path, "r").read().strip()

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        if self.label_transform:
            label = self.label_transform(label)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return (image, label), {}
