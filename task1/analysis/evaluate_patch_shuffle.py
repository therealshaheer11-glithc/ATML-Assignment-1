"""Prepare and evaluate the fixed Task 1 pixel-space patch shuffle."""

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
    make_patch_permutation,
    patch_shuffle,
)


SEED = 6304
N = 500
GRID_SIZE = 4
PATCH_SIZE = 56
BACKBONES = ("resnet50", "vit_b_16", "clip_vit_b_32")
PREDICTORS = {
    "resnet50": ("resnet50_linear_head",),
    "vit_b_16": ("vit_b_16_linear_head",),
    "clip_vit_b_32": (
        "clip_vit_b_32_linear_head",
        "clip_vit_b_32_zero_shot",
    ),
}
MANIFEST_NAME = "patch_shuffle_inputs.csv"
MANIFEST_FIELDS = (
    "official_index",
    "true_class_id",
    "grid_rows",
    "grid_columns",
    "patch_size_pixels",
    "permutation_0_based_json",
    "shuffled_rgb_sha256",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "task1/analysis/evaluate_patch_shuffle.py",
        "task1/analysis/evaluate_clean.py",
        "task1/analysis/evaluate_color.py",
        "task1/models/backbones.py",
        "task1/transforms.py",
    ):
        digest.update((repo_root / relative).read_bytes())
    return digest.hexdigest()


def csv_bytes(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing file differs from fixed protocol: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary file: {temporary}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def save_tensor_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(prior, dict)
            or set(prior) != set(payload)
            or any(
                not torch.equal(prior[key], value)
                if isinstance(value, Tensor)
                else prior[key] != value
                for key, value in payload.items()
            )
        ):
            raise RuntimeError(f"Existing tensor differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary tensor: {temporary}")
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


def rgb_sha256(image: Tensor) -> str:
    if image.shape != (3, 224, 224) or image.dtype != torch.float32:
        raise RuntimeError("Expected one common 224x224 float32 RGB image.")
    if not bool(((image >= 0) & (image <= 1)).all()):
        raise RuntimeError("Patch-shuffled image is outside [0, 1].")
    pixels = image.mul(255).round().to(torch.uint8).contiguous().numpy()
    return hashlib.sha256(pixels.tobytes()).hexdigest()


class PatchShuffledSubset(Dataset):
    def __init__(self, clean_dataset: EvaluationSubset) -> None:
        self.clean_dataset = clean_dataset

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, position: int) -> tuple[Tensor, int, int]:
        image, label, official_index = self.clean_dataset[position]
        permutation = make_patch_permutation(official_index, seed=SEED)
        return patch_shuffle(image, permutation), label, official_index


def prepare_manifest(
    clean_dataset: EvaluationSubset, repo_root: Path, artifacts_root: Path
) -> tuple[list[dict[str, Any]], Path]:
    if len(clean_dataset) != N:
        raise RuntimeError("Patch experiment requires all 500 fixed images.")
    rows: list[dict[str, Any]] = []
    for position in range(N):
        image, label, official_index = clean_dataset[position]
        permutation = make_patch_permutation(official_index, seed=SEED)
        if len(permutation) != 16 or permutation == tuple(range(16)):
            raise RuntimeError("Patch permutation is missing or identity.")
        shuffled = patch_shuffle(image, permutation)
        rows.append({
            "official_index": official_index,
            "true_class_id": label,
            "grid_rows": GRID_SIZE,
            "grid_columns": GRID_SIZE,
            "patch_size_pixels": PATCH_SIZE,
            "permutation_0_based_json": json.dumps(
                permutation, separators=(",", ":")
            ),
            "shuffled_rgb_sha256": rgb_sha256(shuffled),
        })
    manifest_bytes = csv_bytes(rows, MANIFEST_FIELDS)
    artifact_path = artifacts_root / "patch_shuffle_inputs" / MANIFEST_NAME
    repository_path = repo_root / "task1/results/patch_shuffle" / MANIFEST_NAME
    write_once(artifact_path, manifest_bytes)
    write_once(repository_path, manifest_bytes)
    print(
        "PASS: 500 deterministic non-identity 4x4 permutations and "
        "image hashes recorded."
    )
    return rows, artifact_path


def load_or_extract_features(
    backbone: FrozenBackbone,
    dataset: PatchShuffledSubset,
    manifest_rows: list[dict[str, Any]],
    manifest_path: Path,
    repo_root: Path,
    artifacts_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    clean = dataset.clean_dataset
    expected_labels = torch.tensor(clean.labels, dtype=torch.long)
    expected_indices = torch.tensor(clean.indices, dtype=torch.long)
    expected_hashes = [row["shuffled_rgb_sha256"] for row in manifest_rows]
    metadata = {
        "dataset": "STL-10",
        "official_partition": "test",
        "selection_seed": SEED,
        "evaluation_manifest_sha256": sha256_file(clean.manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "condition": "patch_shuffle",
        "grid": [GRID_SIZE, GRID_SIZE],
        "patch_size_pixels": [PATCH_SIZE, PATCH_SIZE],
        "permutations_per_image": 1,
        "non_identity": True,
        "permutation_seed": SEED,
        "backbone": backbone.metadata.key,
        "weights": backbone.metadata.weights_identifier,
        "representation": backbone.metadata.representation,
        "feature_dim": backbone.feature_dim,
        "inference_device": str(device),
        "source_fingerprint": source_fingerprint(repo_root),
        "shuffle": False,
        "gradients": False,
    }
    cache_path = (
        artifacts_root / "features" / backbone.metadata.key / "patch_shuffle.pt"
    )
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
            or not torch.equal(cache.get("labels"), expected_labels)
            or not isinstance(cache.get("official_indices"), Tensor)
            or not torch.equal(cache.get("official_indices"), expected_indices)
            or cache.get("input_sha256") != expected_hashes
        ):
            raise RuntimeError(f"Existing patch cache is invalid: {cache_path}")
        print(f"PASS: reused verified {backbone.metadata.key} patch features")
        return cache

    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("Backbone is not completely frozen.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    index_parts: list[Tensor] = []
    image_hashes: list[str] = []
    with torch.inference_mode():
        for batch_number, (images, labels, indices) in enumerate(loader, 1):
            batch_hashes = [rgb_sha256(image) for image in images]
            start = len(image_hashes)
            if batch_hashes != expected_hashes[start : start + len(batch_hashes)]:
                raise RuntimeError("Shuffled inputs differ from fixed manifest.")
            image_hashes.extend(batch_hashes)
            features = backbone(images.to(device)).float().cpu()
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError("Non-finite patch-shuffled features.")
            feature_parts.append(features)
            label_parts.append(labels.long())
            index_parts.append(indices.long())
            print(
                f"{backbone.metadata.key} patch shuffle: feature batch "
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
        or image_hashes != expected_hashes
    ):
        raise RuntimeError("Extracted patch cache failed validation.")
    save_tensor_once(cache_path, cache)
    print(f"PASS: saved {backbone.metadata.key} patch features")
    return cache


def evaluate_predictor(
    predictor: str,
    backbone: FrozenBackbone,
    cache: dict[str, Any],
    clean_results: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    artifacts_root: Path,
    repo_root: Path,
    device: torch.device,
) -> None:
    metric_path = artifacts_root / "metrics/patch_shuffle" / f"{predictor}.json"
    rows_path = (
        artifacts_root / "metrics/patch_shuffle" / f"{predictor}_predictions.csv"
    )
    tensor_path = artifacts_root / "predictions/patch_shuffle" / f"{predictor}.pt"
    repository_dir = repo_root / "task1/results/patch_shuffle"
    clean_accuracy = float(clean_results[predictor]["top1_accuracy"])
    if all(path.is_file() for path in (metric_path, rows_path, tensor_path)):
        prior_record = json.loads(metric_path.read_text(encoding="utf-8"))
        with rows_path.open(newline="", encoding="utf-8") as file:
            prior_rows = list(csv.DictReader(file))
        prior_tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if (
            prior_record.get("predictor") != predictor
            or prior_record.get("condition") != "patch_shuffle"
            or prior_record.get("number_of_examples") != N
            or prior_record.get("own_clean_accuracy") != clean_accuracy
            or prior_record.get("source_fingerprint") != source_fingerprint(repo_root)
            or prior_record.get("input_manifest_sha256")
            != cache["metadata"]["input_manifest_sha256"]
            or len(prior_rows) != N
            or [int(row["official_index"]) for row in prior_rows]
            != cache["official_indices"].tolist()
            or not isinstance(prior_tensor, dict)
            or prior_tensor.get("predictor") != predictor
            or not isinstance(prior_tensor.get("predictions"), Tensor)
            or not torch.equal(
                prior_tensor["predictions"],
                torch.tensor(
                    [int(row["shuffled_predicted_class_id"]) for row in prior_rows]
                ),
            )
        ):
            raise RuntimeError(f"Existing patch result is invalid: {predictor}")
        write_once(repository_dir / rows_path.name, rows_path.read_bytes())
        write_once(repository_dir / metric_path.name, metric_path.read_bytes())
        print(f"PASS: reused complete {predictor} patch result")
        return
    clean_payload = load_clean_predictions(
        artifacts_root, predictor, cache["labels"],
        cache["official_indices"], clean_accuracy,
    )
    if predictor.endswith("_linear_head"):
        checkpoint_path: Path | None = (
            artifacts_root / "checkpoints/linear_heads"
            / f"{backbone.metadata.key}.pt"
        )
        logits, _ = linear_head_logits(
            backbone, cache["features"], checkpoint_path, device
        )
        prompts: list[str] | None = None
    elif predictor == "clip_vit_b_32_zero_shot":
        if not isinstance(backbone, CLIPViTB32Backbone):
            raise RuntimeError("Zero-shot predictor requires CLIP.")
        checkpoint_path = None
        logits, prompts = zero_shot_logits(
            backbone, cache["features"], device
        )
    else:
        raise ValueError(f"Unknown predictor: {predictor}")

    scored = calculate_metrics(logits, cache["labels"])
    predictions = scored["predictions"]
    correct_count = int(scored["correct_count"])
    match_count = int((predictions == clean_payload["predictions"]).sum())
    record = {
        "predictor": predictor,
        "condition": "patch_shuffle",
        "number_of_examples": N,
        "correct_count": correct_count,
        "absolute_accuracy": correct_count / N,
        "own_clean_accuracy": clean_accuracy,
        "accuracy_change_from_own_clean_baseline": (
            correct_count / N - clean_accuracy
        ),
        "prediction_match_count": match_count,
        "prediction_consistency_with_clean": match_count / N,
        "mean_maximum_confidence": scored["mean_maximum_confidence"],
        "backbone_metadata": asdict(backbone.metadata),
        "inference_device": str(device),
        "feature_cache_metadata": cache["metadata"],
        "evaluation_manifest_sha256": cache["metadata"][
            "evaluation_manifest_sha256"
        ],
        "input_manifest_sha256": cache["metadata"]["input_manifest_sha256"],
        "source_fingerprint": source_fingerprint(repo_root),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path else None
        ),
        "zero_shot_prompts": prompts,
    }
    output_rows = []
    for row, label, clean_prediction, prediction, confidence in zip(
        manifest_rows,
        cache["labels"].tolist(),
        clean_payload["predictions"].tolist(),
        predictions.tolist(),
        scored["confidences"].tolist(),
    ):
        output_rows.append({
            "predictor": predictor,
            "official_index": row["official_index"],
            "true_class_id": label,
            "true_class_name": STL10_CLASS_NAMES[label],
            "permutation_0_based_json": row["permutation_0_based_json"],
            "shuffled_rgb_sha256": row["shuffled_rgb_sha256"],
            "clean_predicted_class_id": clean_prediction,
            "shuffled_predicted_class_id": prediction,
            "shuffled_correct": prediction == label,
            "matches_clean_prediction": prediction == clean_prediction,
            "shuffled_maximum_confidence": confidence,
        })
    tensor_payload = {
        "predictor": predictor,
        "condition": "patch_shuffle",
        "official_indices": cache["official_indices"],
        "labels": cache["labels"],
        "input_sha256": cache["input_sha256"],
        "logits": logits,
        "probabilities": scored["probabilities"],
        "predictions": predictions,
        "clean_predictions": clean_payload["predictions"],
    }
    metric_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    row_bytes = csv_bytes(output_rows, tuple(output_rows[0]))
    save_tensor_once(tensor_path, tensor_payload)
    write_once(rows_path, row_bytes)
    write_once(metric_path, metric_bytes)
    write_once(repository_dir / rows_path.name, row_bytes)
    write_once(repository_dir / metric_path.name, metric_bytes)
    print(
        f"PASS: {predictor} patch shuffle: "
        f"accuracy={record['absolute_accuracy']:.4f}, "
        f"delta={record['accuracy_change_from_own_clean_baseline']:+.4f}, "
        f"consistency={record['prediction_consistency_with_clean']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if args.mode == "evaluate" and args.backbone is None:
        raise ValueError("--backbone is required for evaluation.")
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
    manifest_rows, manifest_path = prepare_manifest(
        clean_dataset, repo_root, artifacts_root
    )
    if args.mode == "prepare":
        return

    clean_results = json.loads(
        (artifacts_root / "metrics/clean/clean_metrics.json")
        .read_text(encoding="utf-8")
    )["results"]
    backbone = build_backbone(args.backbone, device=device)
    dataset = PatchShuffledSubset(clean_dataset)
    cache = load_or_extract_features(
        backbone, dataset, manifest_rows, manifest_path,
        repo_root, artifacts_root, device, args.batch_size,
    )
    for predictor in PREDICTORS[args.backbone]:
        evaluate_predictor(
            predictor, backbone, cache, clean_results,
            manifest_rows, artifacts_root, repo_root, device,
        )
    print("PASS: patch-shuffle backbone stage complete.")


if __name__ == "__main__":
    main()
