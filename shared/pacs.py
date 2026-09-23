"""Locked PACS split, preprocessing, and deterministic domain-balanced sampling."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from torchvision.transforms import InterpolationMode


SEED = 6304
SOURCES = ("photo", "art_painting", "cartoon")
TARGET = "sketch"
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# This is the exact previously verified seed-6304 split the student approved for reuse.
EXPECTED_PROTOCOL_SHA256 = "e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74"
EXPECTED_DATASET_FILE_LIST_SHA256 = "559ac63b8df8e07330b97112e5ec4c414b3957585b28d21b4cfecd2181f538e0"
EXPECTED_SPLIT_COUNTS = {
    "photo": {"train": 1336, "validation": 334},
    "art_painting": {"train": 1638, "validation": 410},
    "cartoon": {"train": 1875, "validation": 469},
    "sketch": 3929,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = (
        root,
        root / "images",
        root / "pacs" / "images",
        root / "PACS" / "images",
        root / "kfold",
        root / "PACS" / "kfold",
    )
    for candidate in candidates:
        if all((candidate / domain).is_dir() for domain in (*SOURCES, TARGET)):
            return candidate
    raise FileNotFoundError(
        f"Could not find Photo/Art/Cartoon/Sketch folders below {root}"
    )


def image_paths(domain_root: Path) -> list[Path]:
    paths = sorted(
        path
        for path in domain_root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No readable image candidates found below {domain_root}")
    return paths


def validate_protocol(protocol: dict) -> None:
    required_header = {
        "dataset": "PACS",
        "seed": SEED,
        "sources": list(SOURCES),
        "target": TARGET,
        "classes": list(CLASSES),
        "file_list_sha256": EXPECTED_DATASET_FILE_LIST_SHA256,
    }
    for key, expected in required_header.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Protocol field {key!r} does not match the locked value")

    seen: set[str] = set()
    for domain in SOURCES:
        domain_splits = protocol.get("source_splits", {}).get(domain, {})
        for split_name in ("train", "validation"):
            rows = domain_splits.get(split_name, [])
            expected_count = EXPECTED_SPLIT_COUNTS[domain][split_name]
            if len(rows) != expected_count:
                raise ValueError(
                    f"Unexpected {domain}/{split_name} count: {len(rows)} != {expected_count}"
                )
            class_counts = [0] * len(CLASSES)
            for row in rows:
                if set(row) != {"path", "class_id"}:
                    raise ValueError(f"Unexpected source fields in {domain}/{split_name}")
                path = row["path"]
                label = row["class_id"]
                if not path.startswith(domain + "/") or not 0 <= label < len(CLASSES):
                    raise ValueError(f"Invalid source record: {row}")
                if path in seen:
                    raise ValueError(f"Duplicate split membership: {path}")
                seen.add(path)
                class_counts[label] += 1
            if any(count == 0 for count in class_counts):
                raise ValueError(f"A class is missing from {domain}/{split_name}")

    target_rows = protocol.get("target_unlabeled", [])
    if len(target_rows) != EXPECTED_SPLIT_COUNTS[TARGET]:
        raise ValueError("Unexpected target-unlabeled count")
    for row in target_rows:
        # No class_id or parsed class name is permitted in a training record.
        if set(row) != {"path", "id"} or not row["path"].startswith(TARGET + "/"):
            raise ValueError("A target training record contains label-like fields")
        if row["path"] in seen:
            raise ValueError(f"Target/source overlap: {row['path']}")
        seen.add(row["path"])


def load_protocol(path: Path) -> dict:
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"Split manifest hash changed: {actual_hash} != {EXPECTED_PROTOCOL_SHA256}"
        )
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def verify_dataset_snapshot(root: Path, protocol: dict) -> dict[str, int | str]:
    resolved = resolve_root(root)
    digest = hashlib.sha256()
    counts: dict[str, int | str] = {}
    unreadable: list[str] = []
    for domain in (*SOURCES, TARGET):
        files = image_paths(resolved / domain)
        counts[domain] = len(files)
        for path in files:
            relative = path.relative_to(resolved).as_posix()
            digest.update(f"{relative}\t{path.stat().st_size}\n".encode())
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                unreadable.append(relative)
    actual = digest.hexdigest()
    if actual != protocol["file_list_sha256"]:
        raise RuntimeError(
            f"PACS file snapshot changed: {actual} != {protocol['file_list_sha256']}"
        )
    if unreadable:
        raise RuntimeError(f"Unreadable PACS files: {unreadable[:5]}")
    return {"root": str(resolved), "file_list_sha256": actual, **counts}


def image_transform(training: bool) -> transforms.Compose:
    operations: list[object] = [
        transforms.Resize(
            (256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True
        )
    ]
    if training:
        operations.extend(
            [
                transforms.RandomCrop(224, padding=None),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    else:
        operations.append(transforms.CenterCrop(224))
    preset = ResNet18_Weights.IMAGENET1K_V1.transforms()
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=preset.mean, std=preset.std),
        ]
    )
    return transforms.Compose(operations)


class SourceDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], training: bool) -> None:
        self.root = resolve_root(root)
        self.records = records
        self.transform = image_transform(training)
        if not records or any(set(row) != {"path", "class_id"} for row in records):
            raise ValueError("SourceDataset received invalid records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(record["class_id"]), record["path"]


class UnlabeledTargetDataset(Dataset):
    """Return no target class label and never parse a target path's parent directory."""

    def __init__(self, root: Path, records: list[dict]) -> None:
        self.root = resolve_root(root)
        self.records = records
        self.transform = image_transform(training=True)
        if not records or any(set(row) != {"path", "id"} for row in records):
            raise ValueError("UnlabeledTargetDataset received label-bearing records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record["id"]


class CyclingBatchSampler(Sampler[list[int]]):
    def __init__(self, count: int, batch_size: int, steps: int, seed: int) -> None:
        if min(count, batch_size, steps) <= 0:
            raise ValueError("Sampler dimensions must be positive")
        self.count = count
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(self.count)
        cursor = 0
        for _ in range(self.steps):
            batch: list[int] = []
            while len(batch) < self.batch_size:
                if cursor == self.count:
                    order = rng.permutation(self.count)
                    cursor = 0
                take = min(self.batch_size - len(batch), self.count - cursor)
                batch.extend(int(i) for i in order[cursor : cursor + take])
                cursor += take
            yield batch


def steps_per_source_epoch(protocol: dict) -> int:
    return max(
        math.ceil(len(protocol["source_splits"][domain]["train"]) / 8)
        for domain in SOURCES
    )


def make_source_train_loader(
    root: Path,
    records: list[dict],
    steps: int,
    sampler_seed: int,
    worker_seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        SourceDataset(root, records, training=True),
        batch_sampler=CyclingBatchSampler(len(records), 8, steps, sampler_seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=np_to_torch_generator(worker_seed),
    )


def make_target_train_loader(
    root: Path,
    records: list[dict],
    steps: int,
    sampler_seed: int,
    worker_seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        UnlabeledTargetDataset(root, records),
        batch_sampler=CyclingBatchSampler(len(records), 24, steps, sampler_seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=np_to_torch_generator(worker_seed),
    )


def make_validation_loader(
    root: Path,
    records: list[dict],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        SourceDataset(root, records, training=False),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def np_to_torch_generator(seed: int):
    # Local import keeps the data-protocol module straightforward to inspect.
    import torch

    return torch.Generator().manual_seed(seed)
