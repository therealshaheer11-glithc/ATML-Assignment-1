"""Evaluation-only CIFAR-100 access.

Training, checkpoint selection, score definition, and threshold calibration must never
import or call this module.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets

from task4.data.cifar10 import evaluation_transform


NEAR_UNKNOWN_CLASSES = (
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel",
)
FAR_UNKNOWN_CLASSES = (
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe",
)


class FixedUnknownSubset(Dataset):
    def __init__(self, base: datasets.CIFAR100, class_names: tuple[str, ...], group: str):
        self.base = base
        self.group = group
        name_to_index = {name: index for index, name in enumerate(base.classes)}
        missing = set(class_names) - set(name_to_index)
        if missing:
            raise ValueError(f"CIFAR-100 classes not found: {sorted(missing)}")
        allowed = {name_to_index[name] for name in class_names}
        self.indices = [i for i, label in enumerate(base.targets) if label in allowed]
        self.class_names = class_names
        if len(self.indices) != 800:
            raise RuntimeError(f"The PA requires exactly 800 {group} unknown images")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        original_index = self.indices[item]
        image, fine_label = self.base[original_index]
        return image, fine_label, original_index, self.group


def load_fixed_unknowns(root: str | Path, download: bool):
    base = datasets.CIFAR100(
        root=str(root), train=False, transform=evaluation_transform(), download=download
    )
    near = FixedUnknownSubset(base, NEAR_UNKNOWN_CLASSES, "near")
    far = FixedUnknownSubset(base, FAR_UNKNOWN_CLASSES, "far")
    return near, far

