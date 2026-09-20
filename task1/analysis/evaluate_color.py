"""Evaluate grayscale and +90-degree hue interventions for Task 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from task1.analysis.evaluate_clean import (  # noqa: E402
    EVALUATION_SIZE,
    EXAMPLES_PER_CLASS,
    FEATURE_BATCH_SIZE,
    PROMPT_TEMPLATE,
    EvaluationSubset,
    calculate_metrics,
    linear_head_logits,
    sha256_file,
    zero_shot_logits,
)
from task1.data.make_subset import STL10_CLASS_NAMES  # noqa: E402
from task1.models.backbones import (  # noqa: E402
    CLIPViTB32Backbone,
    FrozenBackbone,
    build_backbone,
)
from task1.transforms import grayscale, hue_rotate  # noqa: E402


SEED = 6304
CONDITIONS = ("grayscale", "hue_rotation_90")
BACKBONE_NAMES = ("resnet50", "vit_b_16", "clip_vit_b_32")
PREDICTOR_NAMES = (
    "resnet50_linear_head",
    "vit_b_16_linear_head",
    "clip_vit_b_32_linear_head",
    "clip_vit_b_32_zero_shot",
)


class ColorInterventionSubset(Dataset):
    """Apply one locked color intervention after common-image construction."""

    def __init__(self, clean_dataset: EvaluationSubset, condition: str) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown color condition: {condition}")
        self.clean_dataset = clean_dataset
        self.condition = condition

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, position: int) -> tuple[Tensor, int, int]:
        image, label, official_index = self.clean_dataset[position]

        if self.condition == "grayscale":
            transformed = grayscale(image)
        else:
            transformed = hue_rotate(image, angle_degrees=90.0)

        return transformed, label, official_index


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
        "task1/analysis/evaluate_clean.py",
        "task1/analysis/evaluate_color.py",
    ):
        digest.update((repo_root / relative_path).read_bytes())
    return digest.hexdigest()


def cache_is_valid(
    payload: Any,
    expected_metadata: dict[str, Any],
    clean_dataset: EvaluationSubset,
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
        == (len(clean_dataset), expected_metadata["feature_dim"])
        and bool(torch.isfinite(features).all())
        and torch.equal(labels, torch.tensor(clean_dataset.labels))
        and torch.equal(indices, torch.tensor(clean_dataset.indices))
    )


def extract_or_load_features(
    backbone: FrozenBackbone,
    clean_dataset: EvaluationSubset,
    condition: str,
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
        "condition": condition,
        "intervention": (
            "three-channel grayscale"
            if condition == "grayscale"
            else "HSV hue rotation of +90 degrees"
        ),
        "backbone": backbone.metadata.key,
        "weights": backbone.metadata.weights_identifier,
        "representation": backbone.metadata.representation,
        "feature_dim": backbone.feature_dim,
        "common_image_construction": (
            "RGB, direct bicubic resize to 224x224, [0,1]"
        ),
        "intervention_before_model_normalization": True,
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "shuffle": False,
        "gradients": False,
        "source_fingerprint": source_fingerprint(repo_root),
        "manifest_sha256": sha256_file(clean_dataset.manifest_path),
    }

    if cache_path.exists() and not force:
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache_is_valid(cached, metadata, clean_dataset):
            print(
                f"PASS: reused {backbone.metadata.key} {condition} cache"
            )
            return cached
        print(
            f"Stale {backbone.metadata.key} {condition} cache; recomputing."
        )

    dataset = ColorInterventionSubset(clean_dataset, condition)
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
                    f"Non-finite {condition} features in batch "
                    f"{batch_number}."
                )
            feature_batches.append(features)
            label_batches.append(labels.long())
            index_batches.append(indices.long())

            print(
                f"{backbone.metadata.key} {condition}: feature batch "
                f"{batch_number}/{len(loader)}"
            )

    payload = {
        "metadata": metadata,
        "features": torch.cat(feature_batches),
        "labels": torch.cat(label_batches),
        "official_indices": torch.cat(index_batches),
    }
    if not cache_is_valid(payload, metadata, clean_dataset):
        raise RuntimeError(f"Extracted {condition} cache failed validation.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"PASS: saved {backbone.metadata.key} {condition} cache")
    return payload


def load_clean_predictions(
    artifacts_root: Path,
    predictor: str,
    labels: Tensor,
    indices: Tensor,
    clean_accuracy: float,
) -> dict[str, Any]:
    path = artifacts_root / "predictions/clean" / f"{predictor}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)

    if payload["predictor"] != predictor:
        raise RuntimeError("Clean-prediction predictor mismatch.")
    if not torch.equal(payload["labels"], labels):
        raise RuntimeError("Clean and transformed labels differ.")
    if not torch.equal(payload["official_indices"], indices):
        raise RuntimeError("Clean and transformed image indices differ.")

    observed_accuracy = float(
        (payload["predictions"] == labels).float().mean()
    )
    if abs(observed_accuracy - clean_accuracy) > 1e-12:
        raise RuntimeError("Saved clean accuracy is inconsistent.")
    return payload


def record_result(
    predictor: str,
    condition: str,
    logits: Tensor,
    labels: Tensor,
    indices: Tensor,
    clean_record: dict[str, Any],
    artifacts_root: Path,
    aggregate_results: dict[str, dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    metrics = calculate_metrics(logits, labels)
    clean_accuracy = float(clean_record["top1_accuracy"])
    clean_payload = load_clean_predictions(
        artifacts_root=artifacts_root,
        predictor=predictor,
        labels=labels,
        indices=indices,
        clean_accuracy=clean_accuracy,
    )
    consistency = float(
        (
            metrics["predictions"] == clean_payload["predictions"]
        ).float().mean()
    )

    result = {
        "number_of_examples": int(labels.shape[0]),
        "correct_count": metrics["correct_count"],
        "absolute_accuracy": metrics["top1_accuracy"],
        "own_clean_accuracy": clean_accuracy,
        "accuracy_change_from_own_clean_baseline": (
            metrics["top1_accuracy"] - clean_accuracy
        ),
        "prediction_consistency_with_clean": consistency,
        "prediction_match_count": int(
            (
                metrics["predictions"]
                == clean_payload["predictions"]
            ).sum()
        ),
    }
    aggregate_results.setdefault(predictor, {})[condition] = result

    for index, label, clean_prediction, transformed_prediction, confidence in zip(
        indices.tolist(),
        labels.tolist(),
        clean_payload["predictions"].tolist(),
        metrics["predictions"].tolist(),
        metrics["confidences"].tolist(),
    ):
        prediction_rows.append(
            {
                "dataset": "STL-10",
                "official_partition": "test",
                "official_index": index,
                "selection_seed": SEED,
                "condition": condition,
                "predictor": predictor,
                "true_class_id": label,
                "true_class_name": STL10_CLASS_NAMES[label],
                "clean_predicted_class_id": clean_prediction,
                "clean_predicted_class_name": STL10_CLASS_NAMES[
                    clean_prediction
                ],
                "transformed_predicted_class_id": transformed_prediction,
                "transformed_predicted_class_name": STL10_CLASS_NAMES[
                    transformed_prediction
                ],
                "transformed_correct": transformed_prediction == label,
                "matches_clean_prediction": (
                    transformed_prediction == clean_prediction
                ),
                "transformed_maximum_confidence": confidence,
            }
        )

    prediction_directory = artifacts_root / "predictions/color"
    prediction_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predictor": predictor,
            "condition": condition,
            "official_indices": indices,
            "labels": labels,
            "logits": logits,
            "probabilities": metrics["probabilities"],
            "predictions": metrics["predictions"],
            "clean_predictions": clean_payload["predictions"],
        },
        prediction_directory / f"{predictor}__{condition}.pt",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result table.")
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
    manifest_path = repo_root / "task1/results/splits/evaluation_subset.csv"
    official_test = STL10(
        root=str(args.data_root.resolve()),
        split="test",
        download=False,
    )
    clean_dataset = EvaluationSubset(official_test, manifest_path)

    clean_record_path = artifacts_root / "metrics/clean/clean_metrics.json"
    clean_record = json.loads(clean_record_path.read_text())
    clean_results = clean_record["results"]
    if tuple(clean_results) != PREDICTOR_NAMES:
        raise RuntimeError("Unexpected clean predictor set or order.")

    aggregate_results: dict[str, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    reference_labels: Tensor | None = None
    reference_indices: Tensor | None = None

    for backbone_name in BACKBONE_NAMES:
        print(f"\nLoading {backbone_name}...")
        backbone = build_backbone(backbone_name, device=device)

        for condition in CONDITIONS:
            cache_path = (
                artifacts_root
                / "features"
                / backbone_name
                / f"evaluation_{condition}.pt"
            )
            cache = extract_or_load_features(
                backbone=backbone,
                clean_dataset=clean_dataset,
                condition=condition,
                cache_path=cache_path,
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
                    raise RuntimeError("Labels differ across color caches.")
                if not torch.equal(
                    reference_indices, cache["official_indices"]
                ):
                    raise RuntimeError("Indices differ across color caches.")

            linear_predictor = f"{backbone_name}_linear_head"
            checkpoint_path = (
                artifacts_root
                / "checkpoints/linear_heads"
                / f"{backbone_name}.pt"
            )
            logits, _ = linear_head_logits(
                backbone=backbone,
                features=cache["features"],
                checkpoint_path=checkpoint_path,
                device=device,
            )
            record_result(
                predictor=linear_predictor,
                condition=condition,
                logits=logits,
                labels=cache["labels"],
                indices=cache["official_indices"],
                clean_record=clean_results[linear_predictor],
                artifacts_root=artifacts_root,
                aggregate_results=aggregate_results,
                prediction_rows=prediction_rows,
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
                expected_prompts = [
                    PROMPT_TEMPLATE.format_map({"class": name})
                    for name in STL10_CLASS_NAMES
                ]
                if prompts != expected_prompts:
                    raise RuntimeError("Zero-shot prompts changed.")
                record_result(
                    predictor=zero_shot_predictor,
                    condition=condition,
                    logits=zero_shot_values,
                    labels=cache["labels"],
                    indices=cache["official_indices"],
                    clean_record=clean_results[zero_shot_predictor],
                    artifacts_root=artifacts_root,
                    aggregate_results=aggregate_results,
                    prediction_rows=prediction_rows,
                )

        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for predictor in PREDICTOR_NAMES:
        if tuple(aggregate_results[predictor]) != CONDITIONS:
            raise RuntimeError(f"Incomplete color results for {predictor}.")

    aggregate_record = {
        "dataset": "STL-10",
        "official_partition": "test",
        "evaluation_subset_size": EVALUATION_SIZE,
        "examples_per_class": EXAMPLES_PER_CLASS,
        "selection_seed": SEED,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "conditions": {
            "grayscale": {
                "description": "three-channel grayscale",
            },
            "hue_rotation_90": {
                "description": "HSV hue rotation of +90 degrees",
            },
        },
        "intervention_before_model_normalization": True,
        "model_selection_using_test_results": False,
        "results": aggregate_results,
    }

    metrics_directory = artifacts_root / "metrics/color"
    metrics_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_directory / "color_metrics.json"
    predictions_path = metrics_directory / "color_predictions.csv"
    metrics_path.write_text(
        json.dumps(aggregate_record, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(predictions_path, prediction_rows)

    repository_directory = repo_root / "task1/results/color"
    repository_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_path, repository_directory / metrics_path.name)
    shutil.copy2(
        predictions_path,
        repository_directory / predictions_path.name,
    )

    print("\nColor-intervention results:")
    for predictor in PREDICTOR_NAMES:
        for condition in CONDITIONS:
            result = aggregate_results[predictor][condition]
            print(
                f"{predictor} | {condition}: "
                f"accuracy={result['absolute_accuracy']:.4f}, "
                f"delta={result['accuracy_change_from_own_clean_baseline']:+.4f}, "
                f"consistency="
                f"{result['prediction_consistency_with_clean']:.4f}"
            )
    print("PASS: color interventions completed and recorded.")


if __name__ == "__main__":
    main()
