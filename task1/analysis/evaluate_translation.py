"""Evaluate Task 1 reflection translations on the fixed STL-10 test subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
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
    EvaluationSubset,
    calculate_metrics,
    linear_head_logits,
    zero_shot_logits,
)
from task1.analysis.evaluate_color import load_clean_predictions  # noqa: E402
from task1.data.make_subset import STL10_CLASS_NAMES  # noqa: E402
from task1.models.backbones import (  # noqa: E402
    CLIPViTB32Backbone,
    FrozenBackbone,
    build_backbone,
)
from task1.transforms import (  # noqa: E402
    TRANSLATION_DIRECTIONS,
    reflection_translate,
)


SEED = 6304
N = 500
DISPLACEMENTS = (8, 16, 32)
BACKBONES = ("resnet50", "vit_b_16", "clip_vit_b_32")
PREDICTORS = {
    "resnet50": ("resnet50_linear_head",),
    "vit_b_16": ("vit_b_16_linear_head",),
    "clip_vit_b_32": (
        "clip_vit_b_32_linear_head",
        "clip_vit_b_32_zero_shot",
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing result differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary result exists: {temporary}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def save_tensor_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite tensor result: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary tensor exists: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
    for relative in (
        "task1/analysis/evaluate_translation.py",
        "task1/analysis/evaluate_clean.py",
        "task1/analysis/evaluate_color.py",
        "task1/models/backbones.py",
        "task1/transforms.py",
    ):
        digest.update((repo_root / relative).read_bytes())
    return digest.hexdigest()


class TranslatedSubset(Dataset):
    def __init__(
        self, clean_dataset: EvaluationSubset, displacement: int, direction: str
    ) -> None:
        if displacement not in DISPLACEMENTS:
            raise ValueError("Expected displacement 8, 16, or 32 pixels.")
        if direction not in TRANSLATION_DIRECTIONS:
            raise ValueError(f"Invalid translation direction: {direction}")
        self.clean_dataset = clean_dataset
        self.displacement = displacement
        self.direction = direction

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, position: int) -> tuple[Tensor, int, int]:
        image, label, official_index = self.clean_dataset[position]
        transformed = reflection_translate(
            image, self.displacement, self.direction
        )
        return transformed, label, official_index


def batch_image_hashes(images: Tensor) -> list[str]:
    if images.dtype != torch.float32 or images.ndim != 4:
        raise RuntimeError("Unexpected common-image batch.")
    pixels = images.mul(255).round().to(torch.uint8).contiguous().numpy()
    return [hashlib.sha256(image.tobytes()).hexdigest() for image in pixels]


def input_manifest_bytes(
    indices: Tensor, image_hashes: list[str], displacement: int, direction: str
) -> bytes:
    return csv_bytes(
        [
            {
                "official_index": index,
                "displacement_pixels": displacement,
                "direction": direction,
                "transformed_rgb_sha256": image_hash,
            }
            for index, image_hash in zip(indices.tolist(), image_hashes)
        ],
        [
            "official_index", "displacement_pixels", "direction",
            "transformed_rgb_sha256",
        ],
    )


def load_or_extract_features(
    backbone: FrozenBackbone,
    dataset: TranslatedSubset,
    repo_root: Path,
    artifacts_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    condition = f"d{dataset.displacement}_{dataset.direction}"
    cache_path = (
        artifacts_root / "features" / backbone.metadata.key
        / f"translation_{condition}.pt"
    )
    clean = dataset.clean_dataset
    expected_labels = torch.tensor(clean.labels, dtype=torch.long)
    expected_indices = torch.tensor(clean.indices, dtype=torch.long)
    metadata = {
        "dataset": "STL-10",
        "official_partition": "test",
        "selection_seed": SEED,
        "evaluation_manifest_sha256": sha256_file(clean.manifest_path),
        "condition": condition,
        "transformation": "reflection padding followed by shifted crop",
        "backbone": backbone.metadata.key,
        "weights": backbone.metadata.weights_identifier,
        "representation": backbone.metadata.representation,
        "feature_dim": backbone.feature_dim,
        "source_fingerprint": source_fingerprint(repo_root),
        "shuffle": False,
        "gradients": False,
    }
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(cache, dict)
            or cache.get("metadata") != metadata
            or not isinstance(cache.get("features"), Tensor)
            or tuple(cache["features"].shape) != (N, backbone.feature_dim)
            or cache["features"].dtype != torch.float32
            or not bool(torch.isfinite(cache["features"]).all())
            or not isinstance(cache.get("labels"), Tensor)
            or not torch.equal(cache["labels"], expected_labels)
            or not isinstance(cache.get("official_indices"), Tensor)
            or not torch.equal(cache["official_indices"], expected_indices)
            or not isinstance(cache.get("input_sha256"), list)
            or len(cache["input_sha256"]) != N
        ):
            raise RuntimeError(f"Existing translation cache is invalid: {cache_path}")
        print(f"PASS: reused verified {backbone.metadata.key} {condition} features")
    else:
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise RuntimeError("Backbone is not completely frozen.")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        feature_parts: list[Tensor] = []
        label_parts: list[Tensor] = []
        index_parts: list[Tensor] = []
        image_hashes: list[str] = []
        with torch.inference_mode():
            for batch_number, (images, labels, indices) in enumerate(loader, 1):
                image_hashes.extend(batch_image_hashes(images))
                features = backbone(images.to(device)).float().cpu()
                if not bool(torch.isfinite(features).all()):
                    raise RuntimeError("Non-finite translated features.")
                feature_parts.append(features)
                label_parts.append(labels.long())
                index_parts.append(indices.long())
                print(
                    f"{backbone.metadata.key} {condition}: feature batch "
                    f"{batch_number}/{len(loader)}",
                    flush=True,
                )
        cache = {
            "metadata": metadata,
            "features": torch.cat(feature_parts),
            "labels": torch.cat(label_parts),
            "official_indices": torch.cat(index_parts),
            "input_sha256": image_hashes,
        }
        if (
            tuple(cache["features"].shape) != (N, backbone.feature_dim)
            or not torch.equal(cache["labels"], expected_labels)
            or not torch.equal(cache["official_indices"], expected_indices)
            or len(image_hashes) != N
        ):
            raise RuntimeError("Extracted translation cache failed validation.")
        save_tensor_once(cache_path, cache)
        print(f"PASS: saved {backbone.metadata.key} {condition} features")

    shared_manifest = input_manifest_bytes(
        cache["official_indices"], cache["input_sha256"],
        dataset.displacement, dataset.direction,
    )
    manifest_name = f"input_hashes_{condition}.csv"
    write_once(
        artifacts_root / "translation_inputs" / manifest_name,
        shared_manifest,
    )
    write_once(
        repo_root / "task1/results/translation" / manifest_name,
        shared_manifest,
    )
    return cache


def evaluate_predictor(
    predictor: str,
    backbone: FrozenBackbone,
    cache: dict[str, Any],
    clean_results: dict[str, Any],
    artifacts_root: Path,
    repo_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    condition = cache["metadata"]["condition"]
    stem = f"{predictor}__{condition}"
    metrics_dir = artifacts_root / "metrics/translation"
    predictions_dir = artifacts_root / "predictions/translation"
    metrics_path = metrics_dir / f"{stem}.json"
    rows_path = metrics_dir / f"{stem}_predictions.csv"
    tensor_path = predictions_dir / f"{stem}.pt"
    existing = [path.exists() for path in (metrics_path, rows_path, tensor_path)]
    if any(existing):
        if not all(existing):
            raise RuntimeError(f"Partial existing result: {stem}")
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            record.get("predictor") != predictor
            or record.get("condition") != condition
            or record.get("number_of_examples") != N
            or record.get("evaluation_manifest_sha256")
            != cache["metadata"]["evaluation_manifest_sha256"]
            or record.get("source_fingerprint") != source_fingerprint(repo_root)
        ):
            raise RuntimeError(f"Existing result is stale: {stem}")
        with rows_path.open(newline="", encoding="utf-8") as file:
            stored_rows = list(csv.DictReader(file))
        stored_tensor = torch.load(
            tensor_path, map_location="cpu", weights_only=True
        )
        if (
            len(stored_rows) != N
            or [int(row["official_index"]) for row in stored_rows]
            != cache["official_indices"].tolist()
            or not isinstance(stored_tensor, dict)
            or stored_tensor.get("predictor") != predictor
            or stored_tensor.get("condition") != condition
            or not torch.equal(
                stored_tensor["predictions"],
                torch.tensor(
                    [int(row["transformed_predicted_class_id"])
                     for row in stored_rows]
                ),
            )
        ):
            raise RuntimeError(f"Existing result evidence is invalid: {stem}")
        for path in (metrics_path, rows_path):
            write_once(
                repo_root / "task1/results/translation" / path.name,
                path.read_bytes(),
            )
        print(f"PASS: reused complete {stem} result")
        return record

    clean_record = clean_results[predictor]
    clean_accuracy = float(clean_record["top1_accuracy"])
    clean_payload = load_clean_predictions(
        artifacts_root, predictor, cache["labels"],
        cache["official_indices"], clean_accuracy,
    )
    prompts: list[str] | None = None
    checkpoint_path: Path | None = None
    if predictor.endswith("_linear_head"):
        checkpoint_path = (
            artifacts_root / "checkpoints/linear_heads"
            / f"{backbone.metadata.key}.pt"
        )
        logits, _ = linear_head_logits(
            backbone, cache["features"], checkpoint_path, device
        )
    elif predictor == "clip_vit_b_32_zero_shot":
        if not isinstance(backbone, CLIPViTB32Backbone):
            raise RuntimeError("Zero-shot predictor requires CLIP.")
        logits, prompts = zero_shot_logits(
            backbone, cache["features"], device
        )
    else:
        raise ValueError(f"Unknown predictor: {predictor}")
    scored = calculate_metrics(logits, cache["labels"])
    predictions = scored["predictions"]
    match_count = int((predictions == clean_payload["predictions"]).sum())
    accuracy = scored["correct_count"] / N
    consistency = match_count / N
    record = {
        "predictor": predictor,
        "condition": condition,
        "displacement_pixels": int(condition.split("_")[0][1:]),
        "direction": condition.split("_")[1],
        "number_of_examples": N,
        "correct_count": scored["correct_count"],
        "absolute_accuracy": accuracy,
        "own_clean_accuracy": clean_accuracy,
        "accuracy_change_from_own_clean_baseline": accuracy - clean_accuracy,
        "prediction_match_count": match_count,
        "prediction_consistency_with_clean": consistency,
        "mean_maximum_confidence": scored["mean_maximum_confidence"],
        "backbone_metadata": asdict(backbone.metadata),
        "inference_device": str(device),
        "feature_cache_metadata": cache["metadata"],
        "evaluation_manifest_sha256": cache["metadata"][
            "evaluation_manifest_sha256"
        ],
        "source_fingerprint": source_fingerprint(repo_root),
        "input_manifest_sha256": sha256_file(
            artifacts_root / "translation_inputs"
            / f"input_hashes_{condition}.csv"
        ),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path else None
        ),
        "zero_shot_prompts": prompts,
    }
    output_rows: list[dict[str, Any]] = []
    for index, label, image_hash, clean_prediction, prediction, confidence in zip(
        cache["official_indices"].tolist(),
        cache["labels"].tolist(),
        cache["input_sha256"],
        clean_payload["predictions"].tolist(),
        predictions.tolist(),
        scored["confidences"].tolist(),
    ):
        output_rows.append({
            "predictor": predictor,
            "official_index": index,
            "true_class_id": label,
            "true_class_name": STL10_CLASS_NAMES[label],
            "displacement_pixels": record["displacement_pixels"],
            "direction": record["direction"],
            "transformed_rgb_sha256": image_hash,
            "clean_predicted_class_id": clean_prediction,
            "transformed_predicted_class_id": prediction,
            "transformed_correct": prediction == label,
            "matches_clean_prediction": prediction == clean_prediction,
            "transformed_maximum_confidence": confidence,
        })
    tensor_payload = {
        "predictor": predictor,
        "condition": condition,
        "official_indices": cache["official_indices"],
        "labels": cache["labels"],
        "input_sha256": cache["input_sha256"],
        "logits": logits,
        "probabilities": scored["probabilities"],
        "predictions": predictions,
        "clean_predictions": clean_payload["predictions"],
    }
    metric_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    rows_bytes = csv_bytes(output_rows, list(output_rows[0]))
    save_tensor_once(tensor_path, tensor_payload)
    write_once(rows_path, rows_bytes)
    write_once(metrics_path, metric_bytes)
    repository_dir = repo_root / "task1/results/translation"
    write_once(repository_dir / rows_path.name, rows_bytes)
    write_once(repository_dir / metrics_path.name, metric_bytes)
    print(
        f"PASS: {stem}: accuracy={accuracy:.4f}, "
        f"delta={accuracy - clean_accuracy:+.4f}, "
        f"consistency={consistency:.4f}"
    )
    return record


def write_displacement_summary(
    predictor: str,
    displacement: int,
    direction_records: dict[str, dict[str, Any]],
    repo_root: Path,
    artifacts_root: Path,
) -> None:
    if set(direction_records) != set(TRANSLATION_DIRECTIONS):
        raise RuntimeError("All four translation directions are required.")
    records = [direction_records[direction] for direction in TRANSLATION_DIRECTIONS]
    if any(record["number_of_examples"] != N for record in records):
        raise RuntimeError("Translation direction lost evaluation images.")
    clean_accuracy = records[0]["own_clean_accuracy"]
    if any(record["own_clean_accuracy"] != clean_accuracy for record in records):
        raise RuntimeError("Own clean baseline differs across directions.")
    summary = {
        "predictor": predictor,
        "displacement_pixels": displacement,
        "directions": list(TRANSLATION_DIRECTIONS),
        "examples_per_direction": N,
        "aggregation": "unweighted mean of four directions",
        "own_clean_accuracy": clean_accuracy,
        "mean_accuracy": sum(r["absolute_accuracy"] for r in records) / 4,
        "mean_accuracy_change_from_own_clean_baseline": sum(
            r["accuracy_change_from_own_clean_baseline"] for r in records
        ) / 4,
        "mean_prediction_consistency_with_clean": sum(
            r["prediction_consistency_with_clean"] for r in records
        ) / 4,
        "direction_results": {
            direction: direction_records[direction]
            for direction in TRANSLATION_DIRECTIONS
        },
    }
    data = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    basename = f"{predictor}__d{displacement}_summary.json"
    write_once(artifacts_root / "metrics/translation" / basename, data)
    write_once(repo_root / "task1/results/translation" / basename, data)
    print(
        f"PASS: {predictor} d{displacement} four-direction mean: "
        f"accuracy={summary['mean_accuracy']:.4f}, "
        f"consistency={summary['mean_prediction_consistency_with_clean']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--displacement", type=int, choices=DISPLACEMENTS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    seed_everything()
    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    official_test = STL10(
        root=str(args.data_root.resolve()), split="test", download=False
    )
    clean_dataset = EvaluationSubset(
        official_test,
        repo_root / "task1/results/splits/evaluation_subset.csv",
    )
    clean_record = json.loads(
        (artifacts_root / "metrics/clean/clean_metrics.json")
        .read_text(encoding="utf-8")
    )["results"]
    backbone = build_backbone(args.backbone, device=device)
    results: dict[str, dict[str, dict[str, Any]]] = {
        predictor: {} for predictor in PREDICTORS[args.backbone]
    }
    for direction in TRANSLATION_DIRECTIONS:
        print(
            f"\n{args.backbone}: {args.displacement}px {direction} on {device}",
            flush=True,
        )
        dataset = TranslatedSubset(
            clean_dataset, args.displacement, direction
        )
        cache = load_or_extract_features(
            backbone, dataset, repo_root, artifacts_root,
            device, args.batch_size,
        )
        for predictor in PREDICTORS[args.backbone]:
            results[predictor][direction] = evaluate_predictor(
                predictor, backbone, cache, clean_record,
                artifacts_root, repo_root, device,
            )
        del cache
    for predictor, direction_results in results.items():
        write_displacement_summary(
            predictor, args.displacement, direction_results,
            repo_root, artifacts_root,
        )
    print("PASS: translation displacement stage complete.")


if __name__ == "__main__":
    main()
