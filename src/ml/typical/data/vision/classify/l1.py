from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms as T

from ...base import BaseDataset


class ImageClassificationDatasetL1(BaseDataset):
    def __init__(
        self,
        root: str,
        classes: List[str],
        split: str = "train",
        transform: Callable = T.ToTensor(),
    ) -> None:
        super().__init__(root, split, transform)
        self.root_dir = Path(root)
        self.classes = classes

        self.image_paths = []
        self.labels = []

        split_dir = self.root_dir / split

        for class_idx, class_name in enumerate(self.classes):
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_file in sorted(class_dir.glob("*")):
                    if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.image_paths.append(img_file)
                        self.labels.append(class_idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return (image, label), {}
