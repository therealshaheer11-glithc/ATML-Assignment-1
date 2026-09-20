"""Extract frozen STL-10 features and train one Task 1 linear head."""


from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import STL10


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from task1.data.make_subset import STL10_CLASS_NAMES  # noqa: E402
from task1.models.backbones import (  # noqa: E402
    SUPPORTED_BACKBONES,
    FrozenBackbone,
    build_backbone,
)
from task1.transforms import common_rgb_tensor  # noqa: E402


SEED = 6304
FEATURE_BATCH_SIZE = 64
HEAD_BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUMBER_OF_CLASSES = 10


class ManifestSubset(Dataset):
    """Return manifest-selected images in their recorded order."""

    def __init__(self, dataset: STL10, manifest: Path) -> None:
        self.dataset = dataset
        with manifest.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        if not rows:
            raise RuntimeError(f"Empty manifest: {manifest}")

        self.indices = [int(row["official_index"]) for row in rows]
        self.labels = [int(row["class_id"]) for row in rows]

        if len(self.indices) != len(set(self.indices)):
            raise RuntimeError(f"Duplicate indices in {manifest}")

        for row, index, label in zip(rows, self.indices, self.labels):
            if row["dataset"] != "STL-10":
                raise RuntimeError(f"Wrong dataset in {manifest}")
            if row["official_partition"] != "train":
                raise RuntimeError(f"Non-training image in {manifest}")
            if int(dataset.labels[index]) != label:
                raise RuntimeError(f"Label mismatch at official index {index}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Tensor, int, int]:
        index = self.indices[position]
        image, label = self.dataset[index]
        if int(label) != self.labels[position]:
            raise RuntimeError(f"Label changed at official index {index}")
        return common_rgb_tensor(image), int(label), index


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in (
        "task1/models/backbones.py",
        "task1/transforms.py",
    ):
        digest.update((repo_root / relative_path).read_bytes())
    return digest.hexdigest()


def cache_is_valid(
    payload: Any,
    expected_metadata: dict[str, Any],
    dataset: ManifestSubset,
) -> bool:
    if not isinstance(payload, dict):
        return False
    features = payload.get("features")
    labels = payload.get("labels")
    indices = payload.get("official_indices")
    expected_labels = torch.tensor(dataset.labels, dtype=torch.long)
    expected_indices = torch.tensor(dataset.indices, dtype=torch.long)

    return (
        payload.get("metadata") == expected_metadata
        and isinstance(features, Tensor)
        and isinstance(labels, Tensor)
        and isinstance(indices, Tensor)
        and features.dtype == torch.float32
        and tuple(features.shape)
        == (len(dataset), expected_metadata["feature_dim"])
        and bool(torch.isfinite(features).all())
        and torch.equal(labels, expected_labels)
        and torch.equal(indices, expected_indices)
    )


def extract_or_load_features(
    backbone: FrozenBackbone,
    dataset: ManifestSubset,
    split_name: str,
    cache_path: Path,
    repo_root: Path,
    device: torch.device,
    num_workers: int,
    force: bool,
) -> dict[str, Any]:
    metadata = {
        "dataset": "STL-10",
        "official_partition": "train",
        "split": split_name,
        "seed": SEED,
        "backbone": backbone.metadata.key,
        "weights": backbone.metadata.weights_identifier,
        "representation": backbone.metadata.representation,
        "feature_dim": backbone.feature_dim,
        "image_construction": "RGB, direct bicubic resize to 224x224, [0,1]",
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "shuffle": False,
        "gradients": False,
        "source_fingerprint": source_fingerprint(repo_root),
    }

    if cache_path.exists() and not force:
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache_is_valid(cached, metadata, dataset):
            print(f"PASS: reused verified {split_name} feature cache")
            return cached
        print(f"Existing {split_name} cache is stale; recomputing it.")

    loader = DataLoader(
        dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feature_batches = []
    label_batches = []
    index_batches = []

    backbone.eval()
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("Backbone is not completely frozen.")

    with torch.inference_mode():
        for batch_number, (images, labels, indices) in enumerate(loader, 1):
            features = backbone(
                images.to(device, non_blocking=True)
            ).float().cpu()
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError(f"Non-finite features in batch {batch_number}")
            feature_batches.append(features)
            label_batches.append(labels.long())
            index_batches.append(indices.long())

            if batch_number % 10 == 0 or batch_number == len(loader):
                print(
                    f"{split_name}: feature batch "
                    f"{batch_number}/{len(loader)}"
                )

    payload = {
        "metadata": metadata,
        "features": torch.cat(feature_batches),
        "labels": torch.cat(label_batches),
        "official_indices": torch.cat(index_batches),
    }
    if not cache_is_valid(payload, metadata, dataset):
        raise RuntimeError(f"{split_name} feature cache failed validation.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"PASS: saved verified {split_name} feature cache")
    return payload


def evaluate(
    head: nn.Linear,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    head.eval()
    loss_sum = 0.0
    correct = 0
    count = 0

    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = head(features)
            loss = loss_function(logits, labels)
            batch_size = labels.shape[0]
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(1) == labels).sum().item()
            count += batch_size

    return loss_sum / count, correct / count


def save_history(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_head(
    backbone_name: str,
    backbone_metadata: dict[str, Any],
    train_cache: dict[str, Any],
    validation_cache: dict[str, Any],
    artifacts_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything()
    train_features = train_cache["features"]
    train_labels = train_cache["labels"]
    validation_features = validation_cache["features"]
    validation_labels = validation_cache["labels"]
    feature_dim = train_features.shape[1]

    if validation_features.shape[1] != feature_dim:
        raise RuntimeError("Train/validation feature dimensions differ.")

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=HEAD_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_labels),
        batch_size=HEAD_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    head = nn.Linear(feature_dim, NUMBER_OF_CLASSES).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    configuration = {
        "seed": SEED,
        "head": "Linear",
        "loss": "cross_entropy",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": HEAD_BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "scheduler": None,
        "early_stopping_metric": "validation_accuracy",
        "early_stopping_patience": PATIENCE,
        "checkpoint_rule": "highest validation accuracy; earliest tie",
        "training_shuffle": True,
    }
    checkpoint_path = (
        artifacts_root / "checkpoints/linear_heads" / f"{backbone_name}.pt"
    )
    history_path = (
        artifacts_root / "metrics/training" / f"{backbone_name}_history.csv"
    )
    summary_path = (
        artifacts_root / "metrics/training" / f"{backbone_name}_summary.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        head.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.detach().argmax(1) == labels).sum().item()
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        train_accuracy = train_correct / train_count
        validation_loss, validation_accuracy = evaluate(
            head, validation_loader, loss_function, device
        )
        improved = validation_accuracy > best_accuracy

        if improved:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "backbone": backbone_name,
                    "backbone_metadata": backbone_metadata,
                    "class_names": tuple(STL10_CLASS_NAMES),
                    "training_configuration": configuration,
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_accuracy,
                    "state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value in head.state_dict().items()
                    },
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "improved": improved,
                "epochs_without_improvement": stale_epochs,
            }
        )
        save_history(history, history_path)
        print(
            f"epoch={epoch:02d} train_acc={train_accuracy:.4f} "
            f"val_acc={validation_accuracy:.4f} "
            f"patience={stale_epochs}/{PATIENCE}"
        )

        if stale_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.to(device)
    best_validation_loss, verified_accuracy = evaluate(
        head, validation_loader, loss_function, device
    )
    if abs(verified_accuracy - best_accuracy) > 1e-12:
        raise RuntimeError("Best checkpoint verification failed.")

    summary = {
        "backbone": backbone_name,
        "backbone_metadata": backbone_metadata,
        "class_names": list(STL10_CLASS_NAMES),
        "pytorch_version": str(torch.__version__),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "training_configuration": configuration,
        "data": {
            "training_examples": len(train_features),
            "validation_examples": len(validation_features),
            "official_test_examples_used": 0,
            "unlabeled_examples_used": 0,
        },
        "result": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "best_validation_loss": best_validation_loss,
        },
        "files": {
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=SUPPORTED_BACKBONES)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    device = torch.device(args.device)

    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    seed_everything()
    print(f"Backbone: {args.backbone}; device: {device}")
    print("Only the official labeled STL-10 train partition will be loaded.")

    split_root = repo_root / "task1/results/splits"
    official_train = STL10(
        root=str(args.data_root.resolve()), split="train", download=False
    )
    train_dataset = ManifestSubset(
        official_train, split_root / "head_train_indices.csv"
    )
    validation_dataset = ManifestSubset(
        official_train, split_root / "validation_indices.csv"
    )

    if len(train_dataset) != 4000 or len(validation_dataset) != 1000:
        raise RuntimeError("Expected the preregistered 4000/1000 split.")
    if set(train_dataset.indices) & set(validation_dataset.indices):
        raise RuntimeError("Training and validation splits overlap.")

    backbone = build_backbone(args.backbone, device=device)
    backbone_metadata = asdict(backbone.metadata)
    feature_root = artifacts_root / "features" / args.backbone
    train_cache = extract_or_load_features(
        backbone,
        train_dataset,
        "head_train",
        feature_root / "head_train.pt",
        repo_root,
        device,
        args.num_workers,
        args.force_features,
    )
    validation_cache = extract_or_load_features(
        backbone,
        validation_dataset,
        "validation",
        feature_root / "validation.pt",
        repo_root,
        device,
        args.num_workers,
        args.force_features,
    )

    del backbone, train_dataset, validation_dataset, official_train
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = train_head(
        args.backbone,
        backbone_metadata,
        train_cache,
        validation_cache,
        artifacts_root,
        device,
    )
    print(
        f"PASS: {args.backbone} head trained; "
        f"best epoch={summary['result']['best_epoch']}, "
        f"best validation accuracy="
        f"{summary['result']['best_validation_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
