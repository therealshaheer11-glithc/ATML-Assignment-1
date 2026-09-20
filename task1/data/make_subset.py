
"""Create and record the deterministic STL-10 splits required by Task 1."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10


SEED = 6304

STL10_CLASS_NAMES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_labels(labels: np.ndarray, partition: str) -> None:
    observed = set(map(int, np.unique(labels)))
    expected = set(range(len(STL10_CLASS_NAMES)))

    if observed != expected:
        raise RuntimeError(
            f"Unexpected labels in STL-10 {partition}: "
            f"observed={sorted(observed)}, expected={sorted(expected)}"
        )


def write_manifest(
    path: Path,
    partition: str,
    indices: list[int],
    labels: np.ndarray,
    seed: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "official_partition",
                "official_index",
                "class_id",
                "class_name",
                "selection_seed",
            ],
        )
        writer.writeheader()

        for index in indices:
            class_id = int(labels[index])
            writer.writerow(
                {
                    "dataset": "STL-10",
                    "official_partition": partition,
                    "official_index": index,
                    "class_id": class_id,
                    "class_name": STL10_CLASS_NAMES[class_id],
                    "selection_seed": seed,
                }
            )


def class_counts(indices: list[int], labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(labels[index]) for index in indices)
    return {
        STL10_CLASS_NAMES[class_id]: counts.get(class_id, 0)
        for class_id in range(len(STL10_CLASS_NAMES))
    }


def make_splits(
    data_root: Path,
    output_dir: Path,
    seed: int,
    evaluation_size: int,
    download: bool,
) -> None:
    seed_everything(seed)

    if evaluation_size % len(STL10_CLASS_NAMES) != 0:
        raise ValueError(
            "The evaluation size must be divisible by the 10 STL-10 classes."
        )

    # The assignment permits the official labeled train and test partitions.
    # The separate STL-10 unlabeled partition is deliberately never loaded.
    train_dataset = STL10(
        root=str(data_root),
        split="train",
        download=download,
    )
    test_dataset = STL10(
        root=str(data_root),
        split="test",
        download=download,
    )

    train_labels = np.asarray(train_dataset.labels, dtype=np.int64)
    test_labels = np.asarray(test_dataset.labels, dtype=np.int64)

    validate_labels(train_labels, "train")
    validate_labels(test_labels, "test")

    official_train_indices = np.arange(len(train_dataset), dtype=np.int64)

    head_train_indices, validation_indices = train_test_split(
        official_train_indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=train_labels,
    )

    head_train_indices = sorted(map(int, head_train_indices))
    validation_indices = sorted(map(int, validation_indices))

    # Select the same number of official test examples from every class.
    per_class_target = evaluation_size // len(STL10_CLASS_NAMES)
    rng = np.random.default_rng(seed)
    evaluation_indices: list[int] = []
    insufficient_classes: dict[str, dict[str, int]] = {}

    for class_id, class_name in enumerate(STL10_CLASS_NAMES):
        available_indices = np.flatnonzero(test_labels == class_id)
        number_to_select = min(per_class_target, len(available_indices))

        if number_to_select < per_class_target:
            insufficient_classes[class_name] = {
                "requested": per_class_target,
                "available": int(len(available_indices)),
            }

        chosen = rng.choice(
            available_indices,
            size=number_to_select,
            replace=False,
        )
        evaluation_indices.extend(map(int, chosen))

    evaluation_indices = sorted(evaluation_indices)

    # Integrity checks.
    assert set(head_train_indices).isdisjoint(validation_indices)
    assert sorted(head_train_indices + validation_indices) == list(
        range(len(train_dataset))
    )
    assert len(evaluation_indices) == len(set(evaluation_indices))

    output_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(
        output_dir / "head_train_indices.csv",
        "train",
        head_train_indices,
        train_labels,
        seed,
    )
    write_manifest(
        output_dir / "validation_indices.csv",
        "train",
        validation_indices,
        train_labels,
        seed,
    )
    write_manifest(
        output_dir / "evaluation_subset.csv",
        "test",
        evaluation_indices,
        test_labels,
        seed,
    )

    split_record = {
        "dataset": "STL-10",
        "seed": seed,
        "unlabeled_partition_used": False,
        "split_protocol": {
            "classifier_head_source": "official train partition",
            "classifier_head_train_fraction": 0.80,
            "classifier_head_validation_fraction": 0.20,
            "stratified": True,
            "final_evaluation_source": "official test partition",
            "requested_evaluation_size": evaluation_size,
            "requested_examples_per_class": per_class_target,
        },
        "sizes": {
            "official_train": len(train_dataset),
            "head_train": len(head_train_indices),
            "validation": len(validation_indices),
            "official_test": len(test_dataset),
            "evaluation_subset": len(evaluation_indices),
        },
        "class_counts": {
            "head_train": class_counts(head_train_indices, train_labels),
            "validation": class_counts(validation_indices, train_labels),
            "evaluation_subset": class_counts(
                evaluation_indices,
                test_labels,
            ),
        },
        "insufficient_test_classes": insufficient_classes,
        "indices": {
            "head_train": head_train_indices,
            "validation": validation_indices,
            "evaluation_subset": evaluation_indices,
        },
    }

    with (output_dir / "splits.json").open("w", encoding="utf-8") as file:
        json.dump(split_record, file, indent=2)
        file.write("\n")

    print("STL-10 split generation completed successfully.")
    print(json.dumps(split_record["sizes"], indent=2))
    print("\nEvaluation subset class counts:")
    print(json.dumps(split_record["class_counts"]["evaluation_subset"], indent=2))

    if insufficient_classes:
        print("\nDocumented class insufficiencies:")
        print(json.dumps(insufficient_classes, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--evaluation-size", type=int, default=500)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    make_splits(
        data_root=arguments.data_root,
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        evaluation_size=arguments.evaluation_size,
        download=arguments.download,
    )
