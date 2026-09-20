"""Evaluate all Task 1 clean-baseline predictors on the fixed STL-10 subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from task1.data.make_subset import STL10_CLASS_NAMES  # noqa: E402
from task1.models.backbones import (  # noqa: E402
    CLIPViTB32Backbone,
    FrozenBackbone,
    build_backbone,
)
from task1.transforms import common_rgb_tensor  # noqa: E402


SEED = 6304
FEATURE_BATCH_SIZE = 64
LOGIT_BATCH_SIZE = 256
EVALUATION_SIZE = 500
EXAMPLES_PER_CLASS = 50
PROMPT_TEMPLATE = "a photo of a {class}."
BACKBONE_NAMES = ("resnet50", "vit_b_16", "clip_vit_b_32")


class EvaluationSubset(Dataset):
    """Return the official test images in the saved manifest order."""

    def __init__(self, dataset: STL10, manifest_path: Path) -> None:
        self.dataset = dataset
        self.manifest_path = manifest_path

        with manifest_path.open(newline="", encoding="utf-8") as file:
            self.rows = list(csv.DictReader(file))

        if len(self.rows) != EVALUATION_SIZE:
            raise RuntimeError(
                f"Expected {EVALUATION_SIZE} evaluation rows; "
                f"found {len(self.rows)}."
            )

        self.indices = [int(row["official_index"]) for row in self.rows]
        self.labels = [int(row["class_id"]) for row in self.rows]

        if len(self.indices) != len(set(self.indices)):
            raise RuntimeError("Evaluation manifest contains duplicate indices.")

        expected_counts = {
            class_id: EXAMPLES_PER_CLASS
            for class_id in range(len(STL10_CLASS_NAMES))
        }
        if dict(Counter(self.labels)) != expected_counts:
            raise RuntimeError("Evaluation manifest is not class-balanced.")

        for row, index, label in zip(self.rows, self.indices, self.labels):
            if row["dataset"] != "STL-10":
                raise RuntimeError("Evaluation manifest has the wrong dataset.")
            if row["official_partition"] != "test":
                raise RuntimeError("Evaluation manifest has a non-test row.")
            if int(dataset.labels[index]) != label:
                raise RuntimeError(
                    f"Evaluation label mismatch at official index {index}."
                )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Tensor, int, int]:
        official_index = self.indices[position]
        image, label = self.dataset[official_index]
        expected_label = self.labels[position]

        if int(label) != expected_label:
            raise RuntimeError(
                f"Dataset label changed at official index {official_index}."
            )

        return common_rgb_tensor(image), expected_label, official_index


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    dataset: EvaluationSubset,
) -> bool:
    if not isinstance(payload, dict):
        return False

    features = payload.get("features")
    labels = payload.get("labels")
    indices = payload.get("official_indices")

    return (
        payload.get("metadata") == expected_metadata
        and isinstance(features, Tensor)
        and isinstance(labels, Tensor)
        and isinstance(indices, Tensor)
        and features.dtype == torch.float32
        and tuple(features.shape)
        == (len(dataset), expected_metadata["feature_dim"])
        and bool(torch.isfinite(features).all())
        and torch.equal(labels, torch.tensor(dataset.labels))
        and torch.equal(indices, torch.tensor(dataset.indices))
    )


def extract_or_load_features(
    backbone: FrozenBackbone,
    dataset: EvaluationSubset,
    cache_path: Path,
    repo_root: Path,
    device: torch.device,
    num_workers: int,
    force: bool,
) -> dict[str, Any]:
    metadata = {
        "dataset": "STL-10",
        "official_partition": "test",
        "split": "balanced_evaluation_subset",
        "selection_seed": SEED,
        "size": EVALUATION_SIZE,
        "examples_per_class": EXAMPLES_PER_CLASS,
        "backbone": backbone.metadata.key,
        "weights": backbone.metadata.weights_identifier,
        "representation": backbone.metadata.representation,
        "feature_dim": backbone.feature_dim,
        "image_construction": "RGB, direct bicubic resize to 224x224, [0,1]",
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "shuffle": False,
        "gradients": False,
        "source_fingerprint": source_fingerprint(repo_root),
        "manifest_sha256": sha256_file(dataset.manifest_path),
    }

    if cache_path.exists() and not force:
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache_is_valid(cached, metadata, dataset):
            print(f"PASS: reused verified {backbone.metadata.key} clean cache")
            return cached
        print(f"Stale {backbone.metadata.key} clean cache; recomputing.")

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
                raise RuntimeError(
                    f"Non-finite features in clean batch {batch_number}."
                )

            feature_batches.append(features)
            label_batches.append(labels.long())
            index_batches.append(indices.long())
            print(
                f"{backbone.metadata.key}: clean feature batch "
                f"{batch_number}/{len(loader)}"
            )

    payload = {
        "metadata": metadata,
        "features": torch.cat(feature_batches),
        "labels": torch.cat(label_batches),
        "official_indices": torch.cat(index_batches),
    }

    if not cache_is_valid(payload, metadata, dataset):
        raise RuntimeError("Extracted clean cache failed validation.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"PASS: saved verified {backbone.metadata.key} clean cache")
    return payload


def linear_head_logits(
    backbone: FrozenBackbone,
    features: Tensor,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["backbone"] != backbone.metadata.key:
        raise RuntimeError("Linear-head checkpoint/backbone mismatch.")

    head = nn.Linear(backbone.feature_dim, len(STL10_CLASS_NAMES))
    head.load_state_dict(checkpoint["state_dict"])
    head.to(device).eval()

    logit_batches = []
    with torch.inference_mode():
        for batch in features.split(LOGIT_BATCH_SIZE):
            logit_batches.append(head(batch.to(device)).float().cpu())

    del head
    return torch.cat(logit_batches), checkpoint


def zero_shot_logits(
    backbone: CLIPViTB32Backbone,
    features: Tensor,
    device: torch.device,
) -> tuple[Tensor, list[str]]:
    norms = features.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        raise RuntimeError("CLIP clean image features are not normalized.")

    prompts = [
        PROMPT_TEMPLATE.format_map({"class": class_name})
        for class_name in STL10_CLASS_NAMES
    ]
    text_features = backbone.encode_text(prompts)
    scale = backbone.encoder.logit_scale.exp().detach()

    logit_batches = []
    with torch.inference_mode():
        for batch in features.split(LOGIT_BATCH_SIZE):
            logits = scale * batch.to(device) @ text_features.T
            logit_batches.append(logits.float().cpu())

    return torch.cat(logit_batches), prompts


def calculate_metrics(logits: Tensor, labels: Tensor) -> dict[str, Any]:
    probabilities = logits.softmax(dim=1)
    confidences, predictions = probabilities.max(dim=1)
    labels_numpy = labels.numpy()
    predictions_numpy = predictions.numpy()

    matrix = confusion_matrix(
        labels_numpy,
        predictions_numpy,
        labels=list(range(len(STL10_CLASS_NAMES))),
    )

    return {
        "number_of_examples": int(labels.shape[0]),
        "correct_count": int((predictions == labels).sum()),
        "top1_accuracy": float((predictions == labels).float().mean()),
        "macro_f1": float(
            f1_score(
                labels_numpy,
                predictions_numpy,
                labels=list(range(len(STL10_CLASS_NAMES))),
                average="macro",
                zero_division=0,
            )
        ),
        "mean_maximum_confidence": float(confidences.mean()),
        "predictions": predictions,
        "confidences": confidences,
        "probabilities": probabilities,
        "confusion_matrix": matrix,
    }


def add_prediction_rows(
    output_rows: list[dict[str, Any]],
    predictor: str,
    indices: Tensor,
    labels: Tensor,
    metrics: dict[str, Any],
) -> None:
    for index, label, prediction, confidence in zip(
        indices.tolist(),
        labels.tolist(),
        metrics["predictions"].tolist(),
        metrics["confidences"].tolist(),
    ):
        output_rows.append(
            {
                "dataset": "STL-10",
                "official_partition": "test",
                "official_index": index,
                "selection_seed": SEED,
                "predictor": predictor,
                "true_class_id": label,
                "true_class_name": STL10_CLASS_NAMES[label],
                "predicted_class_id": prediction,
                "predicted_class_name": STL10_CLASS_NAMES[prediction],
                "correct": prediction == label,
                "maximum_confidence": confidence,
            }
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    manifest_path = (
        repo_root / "task1/results/splits/evaluation_subset.csv"
    )
    official_test = STL10(
        root=str(args.data_root.resolve()),
        split="test",
        download=False,
    )
    dataset = EvaluationSubset(official_test, manifest_path)

    aggregate_results: dict[str, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    reference_labels: Tensor | None = None
    reference_indices: Tensor | None = None
    prediction_directory = artifacts_root / "predictions" / "clean"
    prediction_directory.mkdir(parents=True, exist_ok=True)

    for backbone_name in BACKBONE_NAMES:
        print(f"\nEvaluating {backbone_name}...")
        backbone = build_backbone(backbone_name, device=device)
        clean_cache_path = (
            artifacts_root
            / "features"
            / backbone_name
            / "evaluation_clean.pt"
        )
        cache = extract_or_load_features(
            backbone=backbone,
            dataset=dataset,
            cache_path=clean_cache_path,
            repo_root=repo_root,
            device=device,
            num_workers=args.num_workers,
            force=args.force_features,
        )

        if reference_labels is None:
            reference_labels = cache["labels"]
            reference_indices = cache["official_indices"]
        else:
            if not torch.equal(reference_labels, cache["labels"]):
                raise RuntimeError("Labels differ across backbone caches.")
            if not torch.equal(reference_indices, cache["official_indices"]):
                raise RuntimeError("Image indices differ across backbone caches.")

        linear_predictor = f"{backbone_name}_linear_head"
        checkpoint_path = (
            artifacts_root
            / "checkpoints"
            / "linear_heads"
            / f"{backbone_name}.pt"
        )
        logits, checkpoint = linear_head_logits(
            backbone=backbone,
            features=cache["features"],
            checkpoint_path=checkpoint_path,
            device=device,
        )
        metrics = calculate_metrics(logits, cache["labels"])
        aggregate_results[linear_predictor] = {
            "backbone_metadata": asdict(backbone.metadata),
            "head_checkpoint": str(checkpoint_path),
            "head_best_validation_epoch": checkpoint["best_epoch"],
            "head_best_validation_accuracy": checkpoint[
                "best_validation_accuracy"
            ],
            "top1_accuracy": metrics["top1_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mean_maximum_confidence": metrics[
                "mean_maximum_confidence"
            ],
            "correct_count": metrics["correct_count"],
            "number_of_examples": metrics["number_of_examples"],
        }
        add_prediction_rows(
            prediction_rows,
            linear_predictor,
            cache["official_indices"],
            cache["labels"],
            metrics,
        )
        torch.save(
            {
                "predictor": linear_predictor,
                "official_indices": cache["official_indices"],
                "labels": cache["labels"],
                "logits": logits,
                "probabilities": metrics["probabilities"],
                "predictions": metrics["predictions"],
            },
            artifacts_root
            / "predictions"
            / "clean"
            / f"{linear_predictor}.pt",
        )

        for true_id, row in enumerate(metrics["confusion_matrix"]):
            for predicted_id, count in enumerate(row):
                confusion_rows.append(
                    {
                        "predictor": linear_predictor,
                        "true_class_id": true_id,
                        "true_class_name": STL10_CLASS_NAMES[true_id],
                        "predicted_class_id": predicted_id,
                        "predicted_class_name": STL10_CLASS_NAMES[
                            predicted_id
                        ],
                        "count": int(count),
                    }
                )

        if backbone_name == "clip_vit_b_32":
            if not isinstance(backbone, CLIPViTB32Backbone):
                raise TypeError("Expected the CLIP-specific wrapper.")

            zero_shot_predictor = "clip_vit_b_32_zero_shot"
            zero_shot_values, prompts = zero_shot_logits(
                backbone=backbone,
                features=cache["features"],
                device=device,
            )
            zero_shot_metrics = calculate_metrics(
                zero_shot_values,
                cache["labels"],
            )
            aggregate_results[zero_shot_predictor] = {
                "backbone_metadata": asdict(backbone.metadata),
                "prompt_template": PROMPT_TEMPLATE,
                "prompts": prompts,
                "prompt_search_performed": False,
                "similarity_scale": "learned OpenCLIP logit_scale.exp()",
                "top1_accuracy": zero_shot_metrics["top1_accuracy"],
                "macro_f1": zero_shot_metrics["macro_f1"],
                "mean_maximum_confidence": zero_shot_metrics[
                    "mean_maximum_confidence"
                ],
                "correct_count": zero_shot_metrics["correct_count"],
                "number_of_examples": zero_shot_metrics[
                    "number_of_examples"
                ],
            }
            add_prediction_rows(
                prediction_rows,
                zero_shot_predictor,
                cache["official_indices"],
                cache["labels"],
                zero_shot_metrics,
            )
            torch.save(
                {
                    "predictor": zero_shot_predictor,
                    "official_indices": cache["official_indices"],
                    "labels": cache["labels"],
                    "logits": zero_shot_values,
                    "probabilities": zero_shot_metrics["probabilities"],
                    "predictions": zero_shot_metrics["predictions"],
                    "prompts": prompts,
                },
                artifacts_root
                / "predictions"
                / "clean"
                / f"{zero_shot_predictor}.pt",
            )

            for true_id, row in enumerate(
                zero_shot_metrics["confusion_matrix"]
            ):
                for predicted_id, count in enumerate(row):
                    confusion_rows.append(
                        {
                            "predictor": zero_shot_predictor,
                            "true_class_id": true_id,
                            "true_class_name": STL10_CLASS_NAMES[true_id],
                            "predicted_class_id": predicted_id,
                            "predicted_class_name": STL10_CLASS_NAMES[
                                predicted_id
                            ],
                            "count": int(count),
                        }
                    )

        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_directory = artifacts_root / "metrics" / "clean"
    metric_directory.mkdir(parents=True, exist_ok=True)

    aggregate_record = {
        "dataset": "STL-10",
        "official_partition": "test",
        "evaluation_subset_size": EVALUATION_SIZE,
        "examples_per_class": EXAMPLES_PER_CLASS,
        "selection_seed": SEED,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "common_image_construction": (
            "RGB, direct bicubic resize to 224x224, no augmentation"
        ),
        "model_selection_using_test_results": False,
        "results": aggregate_results,
    }

    aggregate_path = metric_directory / "clean_metrics.json"
    predictions_path = metric_directory / "clean_predictions.csv"
    confusion_path = metric_directory / "clean_confusion_matrices.csv"

    aggregate_path.write_text(
        json.dumps(aggregate_record, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(predictions_path, prediction_rows)
    write_csv(confusion_path, confusion_rows)

    repository_result_directory = repo_root / "task1/results/clean"
    repository_result_directory.mkdir(parents=True, exist_ok=True)
    for source in (aggregate_path, predictions_path, confusion_path):
        shutil.copy2(source, repository_result_directory / source.name)

    print("\nClean evaluation results:")
    for predictor, result in aggregate_results.items():
        print(
            f"{predictor}: accuracy={result['top1_accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"mean_confidence={result['mean_maximum_confidence']:.4f}"
        )
    print("PASS: clean evaluation completed and recorded.")


if __name__ == "__main__":
    main()
