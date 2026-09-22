"""Paired representation stability using frozen Task 1 feature caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity


SEED = 6304
N_CLEAN = 500
N_CUE = 200
DIRECTIONS = ("up", "down", "left", "right")
BACKBONES = {
    "resnet50": (2048, ("resnet50_linear_head",)),
    "vit_b_16": (768, ("vit_b_16_linear_head",)),
    "clip_vit_b_32": (
        512,
        ("clip_vit_b_32_linear_head", "clip_vit_b_32_zero_shot"),
    ),
}
ROW_FIELDS = (
    "backbone",
    "predictor",
    "condition",
    "transformed_identifier",
    "clean_official_index",
    "true_class_id",
    "cosine_similarity",
    "clean_predicted_class_id",
    "transformed_predicted_class_id",
    "prediction_consistent",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tensor_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a tensor dictionary: {path}")
    return payload


def check_features(
    payload: dict[str, Any], backbone: str, size: int, width: int,
    path: Path,
) -> Tensor:
    features = payload.get("features")
    metadata = payload.get("metadata")
    if (
        not isinstance(features, Tensor)
        or features.dtype != torch.float32
        or tuple(features.shape) != (size, width)
        or not bool(torch.isfinite(features).all())
        or not isinstance(metadata, dict)
        or metadata.get("backbone") != backbone
        or metadata.get("feature_dim") != width
    ):
        raise RuntimeError(f"Invalid feature cache: {path}")
    return features


def check_predictions(
    payload: dict[str, Any], predictor: str, size: int, path: Path,
) -> Tensor:
    predictions = payload.get("predictions")
    if (
        payload.get("predictor") != predictor
        or not isinstance(predictions, Tensor)
        or tuple(predictions.shape) != (size,)
        or predictions.dtype != torch.int64
        or not bool(((predictions >= 0) & (predictions < 10)).all())
    ):
        raise RuntimeError(f"Invalid prediction cache: {path}")
    return predictions


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ROW_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing analysis differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary file: {temporary}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Cannot summarize an empty condition.")
    cosine = [float(row["cosine_similarity"]) for row in rows]
    stable = [row["prediction_consistent"] == "True" for row in rows]
    stable_cosine = [value for value, match in zip(cosine, stable) if match]
    changed_cosine = [value for value, match in zip(cosine, stable) if not match]
    x_mean = statistics.fmean(cosine)
    y_mean = statistics.fmean(float(value) for value in stable)
    x_centered = [value - x_mean for value in cosine]
    y_centered = [float(value) - y_mean for value in stable]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    correlation = (
        sum(x * y for x, y in zip(x_centered, y_centered)) / denominator
        if denominator > 0 else None
    )
    return {
        "number_of_pairs": len(rows),
        "mean_cosine_similarity": x_mean,
        "median_cosine_similarity": statistics.median(cosine),
        "population_std_cosine_similarity": statistics.pstdev(cosine),
        "prediction_match_count": sum(stable),
        "prediction_consistency_with_clean": y_mean,
        "mean_cosine_when_prediction_stable": (
            statistics.fmean(stable_cosine) if stable_cosine else None
        ),
        "mean_cosine_when_prediction_changed": (
            statistics.fmean(changed_cosine) if changed_cosine else None
        ),
        "pearson_correlation_cosine_vs_prediction_stability": correlation,
    }


def feature_path(root: Path, backbone: str, filename: str) -> Path:
    return root / "features" / backbone / filename


def prediction_path(root: Path, predictor: str, condition: str) -> Path:
    if condition == "clean":
        return root / "predictions/clean" / f"{predictor}.pt"
    if condition == "grayscale":
        return root / "predictions/color" / f"{predictor}__grayscale.pt"
    if condition == "cue_conflict":
        return root / "predictions/cue_conflict" / f"{predictor}.pt"
    if condition == "patch_shuffle":
        return root / "predictions/patch_shuffle" / f"{predictor}.pt"
    return root / "predictions/translation" / f"{predictor}__{condition}.pt"


def condition_cache_name(condition: str) -> str:
    if condition == "grayscale":
        return "evaluation_grayscale.pt"
    if condition == "cue_conflict":
        return "evaluation_cue_conflict.pt"
    if condition == "patch_shuffle":
        return "patch_shuffle.pt"
    return f"translation_{condition}.pt"


def analyze(backbone: str, artifacts_root: Path, repo_root: Path) -> None:
    width, predictors = BACKBONES[backbone]
    clean_path = feature_path(artifacts_root, backbone, "evaluation_clean.pt")
    clean_cache = read_tensor_dict(clean_path)
    clean_features = check_features(
        clean_cache, backbone, N_CLEAN, width, clean_path
    )
    clean_indices = clean_cache.get("official_indices")
    clean_labels = clean_cache.get("labels")
    if (
        not isinstance(clean_indices, Tensor)
        or not isinstance(clean_labels, Tensor)
        or tuple(clean_indices.shape) != (N_CLEAN,)
        or tuple(clean_labels.shape) != (N_CLEAN,)
        or len(set(clean_indices.tolist())) != N_CLEAN
        or Counter(clean_labels.tolist()) != {i: 50 for i in range(10)}
    ):
        raise RuntimeError("Clean feature identities or balance are invalid.")
    index_to_position = {
        index: position for position, index in enumerate(clean_indices.tolist())
    }
    input_hashes = {str(clean_path.relative_to(artifacts_root)): sha256_file(clean_path)}
    clean_predictions: dict[str, Tensor] = {}
    for predictor in predictors:
        path = prediction_path(artifacts_root, predictor, "clean")
        payload = read_tensor_dict(path)
        clean_predictions[predictor] = check_predictions(
            payload, predictor, N_CLEAN, path
        )
        if (
            not torch.equal(payload.get("official_indices"), clean_indices)
            or not torch.equal(payload.get("labels"), clean_labels)
        ):
            raise RuntimeError(f"Clean prediction order differs: {predictor}")
        input_hashes[str(path.relative_to(artifacts_root))] = sha256_file(path)

    conditions = ["grayscale", "cue_conflict", "patch_shuffle"]
    conditions += [
        f"d{distance}_{direction}"
        for distance in (8, 16, 32)
        for direction in DIRECTIONS
    ]
    predictor_rows: dict[str, list[dict[str, Any]]] = {
        predictor: [] for predictor in predictors
    }
    for condition in conditions:
        size = N_CUE if condition == "cue_conflict" else N_CLEAN
        cache_path = feature_path(
            artifacts_root, backbone, condition_cache_name(condition)
        )
        cache = read_tensor_dict(cache_path)
        transformed_features = check_features(
            cache, backbone, size, width, cache_path
        )
        if any(
            cache["metadata"].get(key) != clean_cache["metadata"].get(key)
            for key in ("weights", "representation")
        ):
            raise RuntimeError(f"Backbone weights differ: {condition}")
        if condition == "cue_conflict":
            content_indices = cache.get("content_indices")
            candidate_ids = cache.get("candidate_ids")
            shape_labels = cache.get("shape_labels")
            if (
                not isinstance(content_indices, Tensor)
                or tuple(content_indices.shape) != (N_CUE,)
                or len(set(content_indices.tolist())) != N_CUE
                or not isinstance(candidate_ids, list)
                or len(candidate_ids) != N_CUE
                or len(set(candidate_ids)) != N_CUE
                or not isinstance(shape_labels, Tensor)
                or Counter(shape_labels.tolist()) != {i: 20 for i in range(10)}
            ):
                raise RuntimeError("Cue pairing or class balance is invalid.")
            if any(index not in index_to_position for index in content_indices.tolist()):
                raise RuntimeError("Cue content image missing from clean subset.")
            positions = torch.tensor(
                [index_to_position[index] for index in content_indices.tolist()],
                dtype=torch.long,
            )
            identifiers = candidate_ids
            labels = shape_labels
            if not torch.equal(clean_labels[positions], labels):
                raise RuntimeError("Cue shape labels differ from clean labels.")
        else:
            indices = cache.get("official_indices")
            labels = cache.get("labels")
            if (
                not isinstance(indices, Tensor)
                or not isinstance(labels, Tensor)
                or not torch.equal(indices, clean_indices)
                or not torch.equal(labels, clean_labels)
            ):
                raise RuntimeError(f"Condition order differs: {condition}")
            positions = torch.arange(N_CLEAN)
            identifiers = [str(index) for index in clean_indices.tolist()]

        cosines = cosine_similarity(
            clean_features[positions], transformed_features, dim=1
        )
        if (
            tuple(cosines.shape) != (size,)
            or not bool(torch.isfinite(cosines).all())
            or not bool(((cosines >= -1.000001) & (cosines <= 1.000001)).all())
        ):
            raise RuntimeError(f"Invalid paired cosine values: {condition}")
        input_hashes[str(cache_path.relative_to(artifacts_root))] = sha256_file(
            cache_path
        )
        for predictor in predictors:
            path = prediction_path(artifacts_root, predictor, condition)
            payload = read_tensor_dict(path)
            transformed_predictions = check_predictions(
                payload, predictor, size, path
            )
            if condition == "cue_conflict":
                if (
                    payload.get("candidate_ids") != identifiers
                    or not torch.equal(payload.get("content_indices"),
                                       cache["content_indices"])
                ):
                    raise RuntimeError("Cue prediction order differs.")
            elif not torch.equal(payload.get("official_indices"), clean_indices):
                raise RuntimeError(f"Prediction order differs: {condition}")
            input_hashes[str(path.relative_to(artifacts_root))] = sha256_file(path)
            reference_predictions = clean_predictions[predictor][positions]
            for i in range(size):
                predictor_rows[predictor].append({
                    "backbone": backbone,
                    "predictor": predictor,
                    "condition": condition,
                    "transformed_identifier": identifiers[i],
                    "clean_official_index": int(clean_indices[positions[i]]),
                    "true_class_id": int(labels[i]),
                    "cosine_similarity": float(cosines[i]),
                    "clean_predicted_class_id": int(reference_predictions[i]),
                    "transformed_predicted_class_id": int(
                        transformed_predictions[i]
                    ),
                    "prediction_consistent": str(
                        bool(reference_predictions[i] == transformed_predictions[i])
                    ),
                })
        print(f"PASS: paired {backbone} {condition} features", flush=True)

    for predictor, rows in predictor_rows.items():
        by_condition = {
            condition: summarize(
                [row for row in rows if row["condition"] == condition]
            )
            for condition in conditions
        }
        for distance in (8, 16, 32):
            aggregate_rows = [
                row for row in rows
                if row["condition"].startswith(f"d{distance}_")
            ]
            aggregate = summarize(aggregate_rows)
            if aggregate["number_of_pairs"] != 4 * N_CLEAN:
                raise RuntimeError("Translation aggregate lost paired examples.")
            directional_mean = statistics.fmean(
                by_condition[f"d{distance}_{direction}"][
                    "mean_cosine_similarity"
                ]
                for direction in DIRECTIONS
            )
            if abs(aggregate["mean_cosine_similarity"] - directional_mean) > 1e-7:
                raise RuntimeError("Translation aggregation is inconsistent.")
            by_condition[f"d{distance}_mean_four_directions"] = aggregate
        record = {
            "predictor": predictor,
            "backbone": backbone,
            "representation": clean_cache["metadata"]["representation"],
            "seed": SEED,
            "analysis_source_sha256": sha256_file(Path(__file__)),
            "metric": "paired cosine similarity of frozen pre-head features",
            "cue_clean_counterpart": "content official-test image",
            "translation_aggregation": "mean over four directions",
            "input_sha256": dict(sorted(input_hashes.items())),
            "results": by_condition,
        }
        metric_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
        pair_bytes = csv_bytes(rows)
        metric_name = f"{predictor}.json"
        pair_name = f"{predictor}_pairs.csv"
        for root in (
            artifacts_root / "metrics/feature_similarity",
            repo_root / "task1/results/feature_similarity",
        ):
            write_once(root / metric_name, metric_bytes)
            write_once(root / pair_name, pair_bytes)
        print(
            f"PASS: {predictor}: grayscale="
            f"{by_condition['grayscale']['mean_cosine_similarity']:.4f}, "
            f"cue={by_condition['cue_conflict']['mean_cosine_similarity']:.4f}, "
            f"patch={by_condition['patch_shuffle']['mean_cosine_similarity']:.4f}"
        )
    print(f"PASS: {backbone} representation stability complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=tuple(BACKBONES), required=True)
    args = parser.parse_args()
    analyze(args.backbone, args.artifacts_root.resolve(), args.repo_root.resolve())


if __name__ == "__main__":
    main()
