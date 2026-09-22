"""Evaluate frozen Task 1 models on the visually accepted AdaIN conflicts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from task1.analysis.evaluate_clean import (  # noqa: E402
    linear_head_logits,
    zero_shot_logits,
)
from task1.data.make_subset import STL10_CLASS_NAMES  # noqa: E402
from task1.models.backbones import (  # noqa: E402
    CLIPViTB32Backbone,
    FrozenBackbone,
    build_backbone,
)
from task1.transforms import load_common_rgb_tensor  # noqa: E402


SEED = 6304
TARGET = 200
PER_DIRECTION = 20
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


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


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
            raise RuntimeError(f"Existing result differs; refusing overwrite: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary result exists: {temporary}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def save_tensor_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite tensor result: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary tensor exists: {temporary}")
    torch.save(value, temporary)
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
        "task1/analysis/evaluate_cue_conflict.py",
        "task1/analysis/evaluate_clean.py",
        "task1/models/backbones.py",
        "task1/transforms.py",
    ):
        digest.update((repo_root / relative).read_bytes())
    return digest.hexdigest()


class AcceptedConflicts(Dataset):
    """Return identical generated RGB images for every backbone."""

    def __init__(self, repo_root: Path, artifacts_root: Path) -> None:
        self.artifacts_root = artifacts_root
        self.manifest_path = (
            artifacts_root / "cue_conflict/manifests/accepted_conflicts.csv"
        )
        self.review_path = (
            artifacts_root / "cue_conflict/manifests/final_review_manifest.csv"
        )
        self.record_path = (
            artifacts_root / "cue_conflict/manifests/selection_record.json"
        )
        fields, self.rows = read_csv(self.manifest_path)
        _, review_rows = read_csv(self.review_path)
        record = json.loads(self.record_path.read_text(encoding="utf-8"))
        if not fields or len(self.rows) != TARGET or len(review_rows) != 265:
            raise RuntimeError("Cue-conflict selection is incomplete.")
        if record["accepted_conflicts_sha256"] != sha256_file(self.manifest_path):
            raise RuntimeError("Accepted selection changed after the freeze.")
        if record["final_review_manifest_sha256"] != sha256_file(self.review_path):
            raise RuntimeError("Final visual-review record changed after the freeze.")
        for basename, path in (
            ("accepted_conflicts.csv", self.manifest_path),
            ("final_review_manifest.csv", self.review_path),
            ("selection_record.json", self.record_path),
        ):
            repository_copy = repo_root / "task1/results/cue_conflict" / basename
            if repository_copy.read_bytes() != path.read_bytes():
                raise RuntimeError(f"Repository and Drive copies differ: {basename}")
        if Counter(row["review_status"] for row in review_rows) != {
            "accepted": 200,
            "rejected": 65,
        }:
            raise RuntimeError("Final visual-review counts changed.")
        selected_ids = {row["candidate_id"] for row in self.rows}
        reviewed_accepted = {
            row["candidate_id"] for row in review_rows
            if row["review_status"] == "accepted"
        }
        if len(selected_ids) != TARGET or selected_ids != reviewed_accepted:
            raise RuntimeError("Accepted set does not match manual review.")
        directions = Counter(row["direction"] for row in self.rows)
        if len(directions) != 10 or set(directions.values()) != {PER_DIRECTION}:
            raise RuntimeError("Accepted set is not balanced by direction.")
        _, subset_rows = read_csv(
            repo_root / "task1/results/splits/evaluation_subset.csv"
        )
        subset = {
            int(row["official_index"]): int(row["class_id"])
            for row in subset_rows
        }
        if len(subset) != 500:
            raise RuntimeError("Fixed evaluation subset is missing or malformed.")
        for row in self.rows:
            content_id = int(row["content_class_id"])
            style_id = int(row["style_class_id"])
            if content_id == style_id:
                raise RuntimeError("Conflict classes are identical.")
            if (
                int(row["shape_class_id"]) != content_id
                or int(row["texture_class_id"]) != style_id
                or subset.get(int(row["content_official_index"])) != content_id
                or subset.get(int(row["style_official_index"])) != style_id
                or row["review_status"] != "accepted"
                or row["review_completed_before_model_evaluation"] != "True"
            ):
                raise RuntimeError(f"Invalid accepted row: {row['candidate_id']}")
            path = self.image_path(row)
            if sha256_file(path) != row["output_sha256"]:
                raise RuntimeError(f"Changed candidate image: {path}")
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (224, 224):
                    raise RuntimeError(f"Invalid generated RGB image: {path}")

    def image_path(self, row: dict[str, str]) -> Path:
        return (
            self.artifacts_root / "cue_conflict/candidates"
            / f"{row['candidate_id']}.png"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tensor:
        return load_common_rgb_tensor(self.image_path(self.rows[index]))

    @property
    def candidate_ids(self) -> list[str]:
        return [row["candidate_id"] for row in self.rows]

    @property
    def content_indices(self) -> Tensor:
        return torch.tensor(
            [int(row["content_official_index"]) for row in self.rows],
            dtype=torch.long,
        )

    @property
    def shape_labels(self) -> Tensor:
        return torch.tensor(
            [int(row["shape_class_id"]) for row in self.rows],
            dtype=torch.long,
        )

    @property
    def texture_labels(self) -> Tensor:
        return torch.tensor(
            [int(row["texture_class_id"]) for row in self.rows],
            dtype=torch.long,
        )


def load_or_extract_features(
    backbone: FrozenBackbone,
    dataset: AcceptedConflicts,
    repo_root: Path,
    artifacts_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    cache_path = (
        artifacts_root / "features" / backbone.metadata.key
        / "evaluation_cue_conflict.pt"
    )
    metadata = {
        "backbone": backbone.metadata.key,
        "weights": backbone.metadata.weights_identifier,
        "representation": backbone.metadata.representation,
        "feature_dim": backbone.feature_dim,
        "accepted_manifest_sha256": sha256_file(dataset.manifest_path),
        "source_fingerprint": source_fingerprint(repo_root),
        "image_construction": "common 224x224 RGB tensor in [0,1]",
        "shuffle": False,
        "gradients": False,
    }
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(cache, dict)
            or not isinstance(cache.get("content_indices"), Tensor)
            or not isinstance(cache.get("shape_labels"), Tensor)
            or not isinstance(cache.get("texture_labels"), Tensor)
            or cache.get("metadata") != metadata
            or cache.get("candidate_ids") != dataset.candidate_ids
            or not torch.equal(cache.get("content_indices"), dataset.content_indices)
            or not torch.equal(cache.get("shape_labels"), dataset.shape_labels)
            or not torch.equal(cache.get("texture_labels"), dataset.texture_labels)
            or not isinstance(cache.get("features"), Tensor)
            or cache["features"].dtype != torch.float32
            or tuple(cache["features"].shape) != (TARGET, backbone.feature_dim)
            or not bool(torch.isfinite(cache["features"]).all())
        ):
            raise RuntimeError(f"Existing cue-conflict cache is invalid: {cache_path}")
        print(f"PASS: reused verified {backbone.metadata.key} cue features")
        return cache

    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("Backbone is not frozen.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pieces: list[Tensor] = []
    with torch.inference_mode():
        for batch_number, images in enumerate(loader, 1):
            features = backbone(images.to(device)).float().cpu()
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError("Non-finite cue-conflict features.")
            pieces.append(features)
            print(
                f"{backbone.metadata.key}: cue feature batch "
                f"{batch_number}/{len(loader)}",
                flush=True,
            )
    cache = {
        "metadata": metadata,
        "candidate_ids": dataset.candidate_ids,
        "content_indices": dataset.content_indices,
        "shape_labels": dataset.shape_labels,
        "texture_labels": dataset.texture_labels,
        "features": torch.cat(pieces),
    }
    if tuple(cache["features"].shape) != (TARGET, backbone.feature_dim):
        raise RuntimeError("Cue-conflict feature matrix has the wrong shape.")
    save_tensor_once(cache_path, cache)
    print(f"PASS: saved {backbone.metadata.key} cue features")
    return cache


def count_decisions(categories: list[str]) -> dict[str, float | int | None]:
    counts = Counter(categories)
    total = len(categories)
    shape = counts["shape"]
    texture = counts["texture"]
    other = counts["other"]
    if total != shape + texture + other:
        raise RuntimeError("Prediction categories do not partition the images.")
    decided = shape + texture
    return {
        "number_of_examples": total,
        "shape_decision_count": shape,
        "texture_decision_count": texture,
        "other_decision_count": other,
        "shape_bias_percent": 100.0 * shape / decided if decided else None,
        "coverage_percent": 100.0 * decided / total if total else None,
    }


def evaluate_predictor(
    predictor: str,
    backbone: FrozenBackbone,
    cache: dict[str, Any],
    dataset: AcceptedConflicts,
    artifacts_root: Path,
    repo_root: Path,
    device: torch.device,
) -> None:
    metrics_dir = artifacts_root / "metrics/cue_conflict"
    predictions_dir = artifacts_root / "predictions/cue_conflict"
    metrics_path = metrics_dir / f"{predictor}.json"
    rows_path = metrics_dir / f"{predictor}_predictions.csv"
    tensor_path = predictions_dir / f"{predictor}.pt"
    existing = [path.exists() for path in (metrics_path, rows_path, tensor_path)]
    if any(existing):
        if not all(existing):
            raise RuntimeError(f"Partial existing result for {predictor}; stop.")
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        _, stored_rows = read_csv(rows_path)
        stored_tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if (
            record.get("accepted_manifest_sha256")
            != sha256_file(dataset.manifest_path)
            or record.get("final_review_manifest_sha256")
            != sha256_file(dataset.review_path)
            or record.get("source_fingerprint") != source_fingerprint(repo_root)
            or record.get("number_of_examples") != TARGET
            or record.get("predictor") != predictor
            or len(stored_rows) != TARGET
            or [row["candidate_id"] for row in stored_rows]
            != dataset.candidate_ids
            or not isinstance(stored_tensor, dict)
            or stored_tensor.get("candidate_ids") != dataset.candidate_ids
            or stored_tensor.get("accepted_manifest_sha256")
            != record.get("accepted_manifest_sha256")
            or not isinstance(stored_tensor.get("predictions"), Tensor)
            or tuple(stored_tensor["predictions"].shape) != (TARGET,)
            or [int(row["predicted_class_id"]) for row in stored_rows]
            != stored_tensor["predictions"].tolist()
        ):
            raise RuntimeError(f"Existing predictor result differs: {predictor}")
        if predictor.endswith("_linear_head"):
            checkpoint_path = (
                artifacts_root / "checkpoints/linear_heads"
                / f"{backbone.metadata.key}.pt"
            )
            if record.get("checkpoint_sha256") != sha256_file(checkpoint_path):
                raise RuntimeError(f"Linear-head checkpoint changed: {predictor}")
        for path in (metrics_path, rows_path):
            repository_copy = (
                repo_root / "task1/results/cue_conflict" / path.name
            )
            write_once(repository_copy, path.read_bytes())
        print(f"PASS: reused complete {predictor} result")
        return

    checkpoint_path: Path | None = None
    prompts: list[str] | None = None
    if predictor.endswith("_linear_head"):
        checkpoint_path = (
            artifacts_root / "checkpoints/linear_heads"
            / f"{backbone.metadata.key}.pt"
        )
        logits, checkpoint = linear_head_logits(
            backbone, cache["features"], checkpoint_path, device
        )
        best_epoch = checkpoint["best_epoch"]
        best_accuracy = checkpoint["best_validation_accuracy"]
    elif predictor == "clip_vit_b_32_zero_shot":
        if not isinstance(backbone, CLIPViTB32Backbone):
            raise RuntimeError("Zero-shot predictor requires the CLIP backbone.")
        logits, prompts = zero_shot_logits(
            backbone, cache["features"], device
        )
        best_epoch = None
        best_accuracy = None
    else:
        raise ValueError(f"Unknown predictor: {predictor}")

    if tuple(logits.shape) != (TARGET, len(STL10_CLASS_NAMES)):
        raise RuntimeError("Cue-conflict logits have the wrong shape.")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Cue-conflict logits are non-finite.")
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    shape_labels = cache["shape_labels"]
    texture_labels = cache["texture_labels"]
    categories = [
        "shape" if prediction == shape else
        "texture" if prediction == texture else "other"
        for prediction, shape, texture in zip(
            predictions.tolist(),
            shape_labels.tolist(),
            texture_labels.tolist(),
        )
    ]
    if len(categories) != TARGET:
        raise RuntimeError("Wrong number of categorized predictions.")

    output_rows: list[dict[str, Any]] = []
    by_direction: dict[str, list[str]] = defaultdict(list)
    by_pair: dict[str, list[str]] = defaultdict(list)
    for row, prediction, probability, category in zip(
        dataset.rows,
        predictions.tolist(),
        confidence.tolist(),
        categories,
    ):
        by_direction[row["direction"]].append(category)
        by_pair[row["pair_name"]].append(category)
        output_rows.append({
            "predictor": predictor,
            "candidate_id": row["candidate_id"],
            "pair_name": row["pair_name"],
            "direction": row["direction"],
            "schedule_rank": row["schedule_rank"],
            "content_official_index": row["content_official_index"],
            "style_official_index": row["style_official_index"],
            "shape_class_id": row["shape_class_id"],
            "shape_class_name": row["shape_class_name"],
            "texture_class_id": row["texture_class_id"],
            "texture_class_name": row["texture_class_name"],
            "predicted_class_id": prediction,
            "predicted_class_name": STL10_CLASS_NAMES[prediction],
            "decision": category,
            "maximum_confidence": probability,
            "output_sha256": row["output_sha256"],
        })
    overall = count_decisions(categories)
    if any(count != PER_DIRECTION for count in map(len, by_direction.values())):
        raise RuntimeError("Direction count changed during scoring.")
    if len(by_direction) != 10 or len(by_pair) != 5:
        raise RuntimeError("Pair or direction count changed during scoring.")
    metrics: dict[str, Any] = {
        "predictor": predictor,
        "backbone_metadata": asdict(backbone.metadata),
        "inference_device": str(device),
        "accepted_manifest_sha256": sha256_file(dataset.manifest_path),
        "final_review_manifest_sha256": sha256_file(dataset.review_path),
        "source_fingerprint": source_fingerprint(repo_root),
        "seed": SEED,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path else None
        ),
        "head_best_validation_epoch": best_epoch,
        "head_best_validation_accuracy": best_accuracy,
        "zero_shot_prompts": prompts,
        "shape_bias_definition": "100 * Nshape / (Nshape + Ntexture)",
        "coverage_definition": "100 * (Nshape + Ntexture) / Ntotal",
        **overall,
        "by_direction": {
            key: count_decisions(value)
            for key, value in sorted(by_direction.items())
        },
        "by_pair": {
            key: count_decisions(value)
            for key, value in sorted(by_pair.items())
        },
    }
    metric_bytes = (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode()
    prediction_fields = list(output_rows[0])
    prediction_bytes = csv_bytes(output_rows, prediction_fields)
    tensor_payload = {
        "predictor": predictor,
        "accepted_manifest_sha256": metrics["accepted_manifest_sha256"],
        "candidate_ids": dataset.candidate_ids,
        "content_indices": cache["content_indices"],
        "shape_labels": shape_labels,
        "texture_labels": texture_labels,
        "features_cache": str(
            artifacts_root / "features" / backbone.metadata.key
            / "evaluation_cue_conflict.pt"
        ),
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "decision_categories": categories,
    }

    save_tensor_once(tensor_path, tensor_payload)
    write_once(rows_path, prediction_bytes)
    write_once(metrics_path, metric_bytes)
    repository_result_dir = repo_root / "task1/results/cue_conflict"
    write_once(repository_result_dir / rows_path.name, prediction_bytes)
    write_once(repository_result_dir / metrics_path.name, metric_bytes)
    print(
        f"PASS: {predictor}: shape={overall['shape_decision_count']}, "
        f"texture={overall['texture_decision_count']}, "
        f"other={overall['other_decision_count']}, "
        f"bias={overall['shape_bias_percent']}, "
        f"coverage={overall['coverage_percent']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=BACKBONES + ("all",), required=True)
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
    dataset = AcceptedConflicts(repo_root, artifacts_root)
    print("PASS: frozen, balanced 200-image cue selection verified.")
    backbones = BACKBONES if args.backbone == "all" else (args.backbone,)
    for name in backbones:
        print(f"\nEvaluating {name} on {device}...", flush=True)
        backbone = build_backbone(name, device=device)
        cache = load_or_extract_features(
            backbone, dataset, repo_root, artifacts_root, device, args.batch_size
        )
        for predictor in PREDICTORS[name]:
            evaluate_predictor(
                predictor, backbone, cache, dataset,
                artifacts_root, repo_root, device,
            )
        del cache, backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("PASS: requested cue-conflict evaluation stage complete.")


if __name__ == "__main__":
    main()
