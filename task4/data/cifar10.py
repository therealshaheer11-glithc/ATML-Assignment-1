from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def training_transform(method: str) -> transforms.Compose:
    operations: list[object] = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if method == "gcsc":
        operations.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif method not in {"vanilla", "proser"}:
        raise ValueError(f"Unknown Task 4 method: {method}")
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    return transforms.Compose(operations)


def stratified_split_indices(
    labels: Sequence[int], validation_fraction: float = 0.1, seed: int = 6304
) -> tuple[list[int], list[int]]:
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id in sorted(np.unique(labels_array).tolist()):
        class_indices = np.flatnonzero(labels_array == class_id)
        shuffled = rng.permutation(class_indices)
        validation_count = int(round(len(class_indices) * validation_fraction))
        if validation_count <= 0 or validation_count >= len(class_indices):
            raise ValueError("Every class must contribute to both split partitions")
        validation_indices.extend(shuffled[:validation_count].tolist())
        train_indices.extend(shuffled[validation_count:].tolist())
    return sorted(train_indices), sorted(validation_indices)


def write_split_manifest(
    path: str | Path, labels: Sequence[int], train_indices: list[int], validation_indices: list[int]
) -> dict[str, object]:
    path = Path(path)
    manifest: dict[str, object] = {
        "dataset": "CIFAR-10 official training partition",
        "seed": 6304,
        "split": "stratified 90/10",
        "class_names": list(CIFAR10_CLASSES),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_class_counts": {
            name: sum(labels[index] == class_id for index in train_indices)
            for class_id, name in enumerate(CIFAR10_CLASSES)
        },
        "validation_class_counts": {
            name: sum(labels[index] == class_id for index in validation_indices)
            for class_id, name in enumerate(CIFAR10_CLASSES)
        },
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return manifest


def load_split_manifest(path: str | Path) -> dict[str, object]:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("seed") != 6304 or manifest.get("split") != "stratified 90/10":
        raise ValueError("Split manifest does not match the PA protocol")
    return manifest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_cifar10(
    root: str | Path,
    split_manifest: str | Path,
    method: str,
    download: bool,
):
    root = str(root)
    base = datasets.CIFAR10(root=root, train=True, transform=None, download=download)
    split_path = Path(split_manifest)
    if split_path.exists():
        manifest = load_split_manifest(split_path)
    else:
        train_indices, validation_indices = stratified_split_indices(base.targets)
        manifest = write_split_manifest(
            split_path, base.targets, train_indices, validation_indices
        )
    train_base = datasets.CIFAR10(
        root=root, train=True, transform=training_transform(method), download=False
    )
    validation_base = datasets.CIFAR10(
        root=root, train=True, transform=evaluation_transform(), download=False
    )
    test = datasets.CIFAR10(
        root=root, train=False, transform=evaluation_transform(), download=download
    )
    train = Subset(train_base, list(manifest["train_indices"]))
    validation = Subset(validation_base, list(manifest["validation_indices"]))
    return train, validation, test, manifest


def unaugmented_training_subset(root: str | Path, split_manifest: str | Path):
    manifest = load_split_manifest(split_manifest)
    base = datasets.CIFAR10(
        root=str(root), train=True, transform=evaluation_transform(), download=False
    )
    return Subset(base, list(manifest["train_indices"]))

