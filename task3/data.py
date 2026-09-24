"""Target-free PACS protocol, preprocessing, and deterministic source sampling."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from torchvision.transforms import InterpolationMode


SEED = 6304
SOURCES = ("photo", "art_painting", "cartoon")
FORBIDDEN_TARGET = "sketch"
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_SOURCE_PROTOCOL_SHA256 = (
    "626d8517b44ad50c0219adf49e827de6538561386791bed29a9153a589cd6abc"
)
EXPECTED_PARENT_PROTOCOL_SHA256 = (
    "e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74"
)
EXPECTED_PARENT_FILE_LIST_SHA256 = (
    "559ac63b8df8e07330b97112e5ec4c414b3957585b28d21b4cfecd2181f538e0"
)
EXPECTED_PACS_ARCHIVE_SHA256 = (
    "0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102"
)
EXPECTED_SOURCE_SNAPSHOT_SHA256 = (
    "8ded350769ee15739f8420e755e50ff4377068a4f54ab1c0ba39d5b125e658d2"
)
EXPECTED_SPLIT_COUNTS = {
    "photo": {"train": 1336, "validation": 334},
    "art_painting": {"train": 1638, "validation": 410},
    "cartoon": {"train": 1875, "validation": 469},
}
EXPECTED_DOMAIN_COUNTS = {
    domain: counts["train"] + counts["validation"]
    for domain, counts in EXPECTED_SPLIT_COUNTS.items()
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_record(record: dict, domain: str) -> None:
    if set(record) != {"path", "class_id"}:
        raise ValueError(f"Unexpected source-record fields: {record}")

    path = PurePosixPath(str(record["path"]))
    parts = path.parts
    if not parts or parts[0] != domain:
        raise ValueError(f"Source record has the wrong domain: {record}")
    if FORBIDDEN_TARGET in {part.lower() for part in parts}:
        raise RuntimeError(f"Sketch record rejected by Task 3: {record['path']}")

    class_id = record["class_id"]
    if not isinstance(class_id, int) or not 0 <= class_id < len(CLASSES):
        raise ValueError(f"Invalid source class ID: {record}")


def validate_source_protocol(protocol: dict) -> None:
    expected_header = {
        "dataset": "PACS",
        "seed": SEED,
        "sources": list(SOURCES),
        "classes": list(CLASSES),
        "parent_task2_protocol_sha256": EXPECTED_PARENT_PROTOCOL_SHA256,
        "parent_dataset_file_list_sha256": EXPECTED_PARENT_FILE_LIST_SHA256,
        "pacs_archive_sha256": EXPECTED_PACS_ARCHIVE_SHA256,
        "target_domain_embargoed": FORBIDDEN_TARGET,
        "target_records_included": False,
    }
    for name, expected in expected_header.items():
        if protocol.get(name) != expected:
            raise ValueError(
                f"Source protocol field {name!r} differs: "
                f"expected {expected!r}, got {protocol.get(name)!r}"
            )

    forbidden_keys = {"target", "target_unlabeled", "target_splits"}
    present_forbidden = forbidden_keys.intersection(protocol)
    if present_forbidden:
        raise RuntimeError(
            f"Target-bearing fields are forbidden in Task 3: {sorted(present_forbidden)}"
        )

    seen: set[str] = set()
    for domain in SOURCES:
        domain_splits = protocol.get("source_splits", {}).get(domain, {})
        for split_name in ("train", "validation"):
            records = domain_splits.get(split_name, [])
            expected_count = EXPECTED_SPLIT_COUNTS[domain][split_name]
            if len(records) != expected_count:
                raise ValueError(
                    f"Unexpected {domain}/{split_name} count: "
                    f"{len(records)} != {expected_count}"
                )

            class_counts = [0] * len(CLASSES)
            for record in records:
                validate_source_record(record, domain)
                path = record["path"]
                if path in seen:
                    raise ValueError(f"Duplicate source split membership: {path}")
                seen.add(path)
                class_counts[record["class_id"]] += 1

            if any(count == 0 for count in class_counts):
                raise ValueError(f"A class is missing from {domain}/{split_name}")

    expected_total = sum(EXPECTED_DOMAIN_COUNTS.values())
    if len(seen) != expected_total:
        raise ValueError(f"Unexpected unique source record count: {len(seen)}")


def load_source_protocol(path: Path) -> dict:
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_SOURCE_PROTOCOL_SHA256:
        raise RuntimeError(
            "Source-only protocol hash changed: "
            f"{actual_hash} != {EXPECTED_SOURCE_PROTOCOL_SHA256}"
        )
    protocol = json.loads(path.read_text())
    validate_source_protocol(protocol)
    return protocol


def resolve_source_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not all((resolved / domain).is_dir() for domain in SOURCES):
        raise FileNotFoundError(
            f"Could not find all approved source-domain folders below {resolved}"
        )
    return resolved


def verify_source_snapshot(root: Path, protocol: dict) -> dict[str, int | str]:
    """Verify only source files; never enumerate or open a Sketch directory."""
    resolved = resolve_source_root(root)
    expected_paths = {
        row["path"]
        for domain in SOURCES
        for split_name in ("train", "validation")
        for row in protocol["source_splits"][domain][split_name]
    }

    actual_paths: set[str] = set()
    counts: dict[str, int | str] = {}
    unreadable: list[str] = []
    digest = hashlib.sha256()

    for domain in SOURCES:
        files = sorted(
            path
            for path in (resolved / domain).rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        counts[domain] = len(files)

        if len(files) != EXPECTED_DOMAIN_COUNTS[domain]:
            raise RuntimeError(
                f"Unexpected {domain} source-image count: "
                f"{len(files)} != {EXPECTED_DOMAIN_COUNTS[domain]}"
            )

        for path in files:
            relative = path.relative_to(resolved).as_posix()
            actual_paths.add(relative)
            digest.update(f"{relative}\t{path.stat().st_size}\n".encode("utf-8"))
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                unreadable.append(relative)

    if unreadable:
        raise RuntimeError(f"Unreadable source images: {unreadable[:5]}")

    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing or unexpected:
        raise RuntimeError(
            "Source workspace differs from the approved source protocol: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    actual_hash = digest.hexdigest()
    if actual_hash != EXPECTED_SOURCE_SNAPSHOT_SHA256:
        raise RuntimeError(
            "Source snapshot hash changed: "
            f"{actual_hash} != {EXPECTED_SOURCE_SNAPSHOT_SHA256}"
        )

    return {
        "root": str(resolved),
        "source_snapshot_sha256": actual_hash,
        "source_image_count": len(actual_paths),
        "sketch_images_accessed": 0,
        **counts,
    }


def image_transform(training: bool) -> transforms.Compose:
    operations: list[object] = [
        transforms.Resize(
            (256, 256),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
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
        self.root = resolve_source_root(root)
        self.records = records
        self.transform = image_transform(training)

        if not records:
            raise ValueError("SourceDataset received no records")

        first_domain = PurePosixPath(str(records[0]["path"])).parts[0]
        if first_domain not in SOURCES:
            raise ValueError("SourceDataset received an unknown domain")
        for record in records:
            validate_source_record(record, first_domain)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        relative = PurePosixPath(record["path"])
        if relative.parts[0] not in SOURCES:
            raise RuntimeError("A non-source record reached SourceDataset")
        with Image.open(self.root / Path(*relative.parts)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(record["class_id"]), record["path"]


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
                batch.extend(int(index) for index in order[cursor : cursor + take])
                cursor += take
            yield batch


def steps_per_source_epoch(protocol: dict) -> int:
    return max(
        math.ceil(len(protocol["source_splits"][domain]["train"]) / 8)
        for domain in SOURCES
    )


def torch_generator(seed: int):
    import torch

    return torch.Generator().manual_seed(seed)


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
        batch_sampler=CyclingBatchSampler(
            len(records),
            batch_size=8,
            steps=steps,
            seed=sampler_seed,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch_generator(worker_seed),
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

