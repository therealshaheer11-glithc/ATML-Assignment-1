"""The fixed PACS source split and label-blind Sketch image loader."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights


SEED = 6304
SOURCES = ("photo", "art_painting", "cartoon")
TARGET = "sketch"
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_root(root: Path) -> Path:
    """Allow either a PACS domain directory or its common kfold parent."""
    root = root.expanduser().resolve()
    for candidate in (
        root, root / "kfold", root / "PACS", root / "PACS" / "kfold",
        root / "images", root / "pacs" / "images", root / "PACS" / "images",
    ):
        if all((candidate / domain).is_dir() for domain in (*SOURCES, TARGET)):
            return candidate
    raise FileNotFoundError(f"Expected PACS domain folders under {root} (or its PACS/kfold child)")


def image_paths(domain_root: Path) -> list[Path]:
    paths = sorted(path for path in domain_root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No images found under {domain_root}")
    return paths


def make_protocol(root: Path) -> dict:
    """Record source labels and splits; enumerate Sketch without reading its labels."""
    root = resolve_root(root)
    protocol: dict = {
        "dataset": "PACS",
        "seed": SEED,
        "sources": list(SOURCES),
        "target": TARGET,
        "classes": list(CLASSES),
        "source_splits": {},
        "target_unlabeled": [],
    }
    fingerprint = hashlib.sha256()
    for domain in (*SOURCES, TARGET):
        files = image_paths(root / domain)
        for path in files:
            relative = path.relative_to(root).as_posix()
            fingerprint.update(f"{relative}\t{path.stat().st_size}\n".encode())
        if domain == TARGET:
            # Neither class names nor class IDs enter the target training records.
            protocol["target_unlabeled"] = [
                {"path": p.relative_to(root).as_posix(), "id": hashlib.sha256(p.relative_to(root).as_posix().encode()).hexdigest()[:16]}
                for p in files
            ]
            continue
        labels = [p.parent.name for p in files]
        unknown = set(labels) - set(CLASSES)
        if unknown or set(labels) != set(CLASSES):
            raise RuntimeError(f"Unexpected/missing class folders in {domain}: {sorted(unknown)}")
        indices = np.arange(len(files))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=SEED, shuffle=True, stratify=labels
        )
        split = {}
        for name, selected in (("train", train_idx), ("validation", val_idx)):
            split[name] = [
                {"path": files[int(i)].relative_to(root).as_posix(), "class_id": CLASSES.index(labels[int(i)])}
                for i in sorted(selected)
            ]
        protocol["source_splits"][domain] = split
    protocol["file_list_sha256"] = fingerprint.hexdigest()
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict) -> None:
    if protocol.get("seed") != SEED or protocol.get("sources") != list(SOURCES) or protocol.get("target") != TARGET or protocol.get("classes") != list(CLASSES):
        raise ValueError("PACS protocol does not match the assignment constants")
    all_paths = set()
    for domain in SOURCES:
        rows = protocol["source_splits"][domain]
        for split in ("train", "validation"):
            if not rows[split]:
                raise ValueError(f"Empty {domain}/{split} split")
            for row in rows[split]:
                if set(row) != {"path", "class_id"} or not row["path"].startswith(domain + "/") or not 0 <= row["class_id"] < len(CLASSES):
                    raise ValueError(f"Invalid {domain}/{split} record")
                if row["path"] in all_paths:
                    raise ValueError(f"Image appears in multiple splits: {row['path']}")
                all_paths.add(row["path"])
    target = protocol["target_unlabeled"]
    if not target:
        raise ValueError("Empty Sketch set")
    for row in target:
        if set(row) != {"path", "id"} or not row["path"].startswith(TARGET + "/"):
            raise ValueError("Sketch training record contains unexpected fields")
        if row["path"] in all_paths:
            raise ValueError("Sketch overlaps a source split")
        all_paths.add(row["path"])


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def transform(training: bool) -> transforms.Compose:
    steps = [transforms.Resize((256, 256))]
    if training:
        steps += [transforms.RandomCrop(224), transforms.RandomHorizontalFlip()]
    else:
        steps += [transforms.CenterCrop(224)]
    mean, std = ResNet18_Weights.IMAGENET1K_V1.transforms().mean, ResNet18_Weights.IMAGENET1K_V1.transforms().std
    steps += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(steps)


class PACSImages(Dataset):
    """Return labels only for source records; Sketch yields image and opaque ID."""

    def __init__(self, root: Path, records: list[dict], training: bool, labeled: bool) -> None:
        self.root = resolve_root(root)
        self.records = records
        self.labeled = labeled
        self.image_transform = transform(training)
        expected = {"path", "class_id"} if labeled else {"path", "id"}
        if not records or any(set(row) != expected for row in records):
            raise ValueError("Dataset records have invalid fields")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            pixels = self.image_transform(image.convert("RGB"))
        if self.labeled:
            return pixels, int(record["class_id"]), record["path"]
        return pixels, record["id"]


class CyclingBatchSampler(Sampler[list[int]]):
    """Produce exact-size batches while cycling shorter domain sets."""

    def __init__(self, count: int, batch_size: int, steps: int, seed: int) -> None:
        if count <= 0 or batch_size <= 0 or steps <= 0:
            raise ValueError("Batch sampler sizes must be positive")
        self.count, self.batch_size, self.steps, self.seed = count, batch_size, steps, seed

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(self.count)
        position = 0
        for _ in range(self.steps):
            batch = []
            while len(batch) < self.batch_size:
                if position == self.count:
                    order = rng.permutation(self.count)
                    position = 0
                take = min(self.batch_size - len(batch), self.count - position)
                batch.extend(map(int, order[position:position + take]))
                position += take
            yield batch


def steps_per_source_epoch(protocol: dict) -> int:
    return max(math.ceil(len(protocol["source_splits"][domain]["train"]) / 8) for domain in SOURCES)
