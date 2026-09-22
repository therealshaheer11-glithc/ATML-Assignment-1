"""Freeze a paired UMAP subset and plot Task 1 cached representations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


SEED = 6304
SIZE = 200
PER_CLASS = 20
PER_CLASS_DIRECTION = 5
DIRECTIONS = ("up", "down", "left", "right")
CONDITIONS = (
    "clean", "grayscale", "cue_conflict", "translation_32px", "patch_shuffle"
)
BACKBONES = {"resnet50": 2048, "vit_b_16": 768, "clip_vit_b_32": 512}
CLASS_NAMES = (
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
)
SELECTION_FIELDS = (
    "candidate_id", "clean_official_index", "true_class_id",
    "translation_direction",
)
COORDINATE_FIELDS = (
    "backbone", "condition", "condition_point_id", "clean_official_index",
    "true_class_id", "translation_direction", "umap_x", "umap_y",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
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
            raise RuntimeError(f"Existing frozen artifact differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary artifact: {temporary}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_selection(
    repo_root: Path, artifacts_root: Path
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    accepted_path = (
        artifacts_root / "cue_conflict/manifests/accepted_conflicts.csv"
    )
    freeze_record_path = (
        artifacts_root / "cue_conflict/manifests/selection_record.json"
    )
    evaluation_path = (
        repo_root / "task1/results/splits/evaluation_subset.csv"
    )
    freeze_record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    if freeze_record.get("accepted_conflicts_sha256") != sha256_file(accepted_path):
        raise RuntimeError("Accepted cue-conflict selection changed.")
    accepted = read_csv(accepted_path)
    evaluation = read_csv(evaluation_path)
    evaluation_labels = {
        int(row["official_index"]): int(row["class_id"])
        for row in evaluation
    }
    if len(accepted) != SIZE or len(evaluation_labels) != 500:
        raise RuntimeError("Fixed input manifests have unexpected sizes.")
    content_indices = [int(row["content_official_index"]) for row in accepted]
    candidate_ids = [row["candidate_id"] for row in accepted]
    class_ids = [int(row["content_class_id"]) for row in accepted]
    if (
        len(set(content_indices)) != SIZE
        or len(set(candidate_ids)) != SIZE
        or Counter(class_ids) != {class_id: PER_CLASS for class_id in range(10)}
    ):
        raise RuntimeError("Cue content images are not unique and class-balanced.")
    for row, image_index, class_id in zip(
        accepted, content_indices, class_ids
    ):
        if (
            evaluation_labels.get(image_index) != class_id
            or int(row["shape_class_id"]) != class_id
            or row["review_status"] != "accepted"
            or row["review_completed_before_model_evaluation"] != "True"
        ):
            raise RuntimeError("Accepted cue row is inconsistent with clean data.")

    direction_by_index: dict[int, str] = {}
    for class_id in range(10):
        indices = sorted(
            index for index, label in zip(content_indices, class_ids)
            if label == class_id
        )
        generator = np.random.default_rng(
            np.random.SeedSequence([SEED, class_id])
        )
        for rank, position in enumerate(generator.permutation(PER_CLASS)):
            direction_by_index[indices[int(position)]] = DIRECTIONS[
                rank // PER_CLASS_DIRECTION
            ]
    rows = [
        {
            "candidate_id": candidate_id,
            "clean_official_index": image_index,
            "true_class_id": class_id,
            "translation_direction": direction_by_index[image_index],
        }
        for candidate_id, image_index, class_id in zip(
            candidate_ids, content_indices, class_ids
        )
    ]
    for class_id in range(10):
        directions = Counter(
            row["translation_direction"] for row in rows
            if row["true_class_id"] == class_id
        )
        if directions != {direction: PER_CLASS_DIRECTION for direction in DIRECTIONS}:
            raise RuntimeError("Translation directions are not balanced by class.")
    selection_bytes = csv_bytes(rows, SELECTION_FIELDS)
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    record = {
        "seed": SEED,
        "number_of_images": SIZE,
        "classes": 10,
        "images_per_class": PER_CLASS,
        "translation_displacement_pixels": 32,
        "directions": list(DIRECTIONS),
        "images_per_class_per_direction": PER_CLASS_DIRECTION,
        "direction_assignment_rule": (
            "Within each class, sort official test indices, permute positions "
            "with NumPy SeedSequence([6304, class_id]), and assign five "
            "successive images to each direction in up/down/left/right order."
        ),
        "selection_source": (
            "The 200 unique clean content images in the visually accepted "
            "cue-conflict manifest, in accepted-manifest order."
        ),
        "accepted_manifest_sha256": sha256_file(accepted_path),
        "evaluation_manifest_sha256": sha256_file(evaluation_path),
        "umap_selection_sha256": selection_sha,
        "model_predictions_used_for_selection": False,
    }
    record_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    artifact_path = artifacts_root / "representation/umap_selection.csv"
    for root in (
        artifacts_root / "representation",
        repo_root / "task1/results/representation",
    ):
        write_once(root / "umap_selection.csv", selection_bytes)
        write_once(root / "umap_selection.json", record_bytes)
    print(
        "PASS: frozen 200-image UMAP selection, 20/class and "
        "5/class/direction; no model predictions used."
    )
    return rows, artifact_path, record


def read_features(
    path: Path, backbone: str, size: int, width: int
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("metadata"), dict)
        or payload["metadata"].get("backbone") != backbone
        or payload["metadata"].get("feature_dim") != width
        or not isinstance(payload.get("features"), Tensor)
        or payload["features"].dtype != torch.float32
        or tuple(payload["features"].shape) != (size, width)
        or not bool(torch.isfinite(payload["features"]).all())
        or not bool((payload["features"].norm(dim=1) > 0).all())
    ):
        raise RuntimeError(f"Invalid feature cache: {path}")
    return payload


def plot_backbone(
    backbone: str,
    repo_root: Path,
    artifacts_root: Path,
    selection_rows: list[dict[str, Any]],
    selection_path: Path,
) -> None:
    try:
        umap_version = version("umap-learn")
    except Exception as error:
        raise RuntimeError("Install umap-learn==0.5.12 first.") from error
    if umap_version != "0.5.12":
        raise RuntimeError(
            f"Expected umap-learn 0.5.12, found {umap_version}."
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import umap

    width = BACKBONES[backbone]
    cache_root = artifacts_root / "features" / backbone
    input_paths: dict[str, Path] = {
        "clean": cache_root / "evaluation_clean.pt",
        "grayscale": cache_root / "evaluation_grayscale.pt",
        "cue_conflict": cache_root / "evaluation_cue_conflict.pt",
        "patch_shuffle": cache_root / "patch_shuffle.pt",
    }
    for direction in DIRECTIONS:
        input_paths[f"translation_32px_{direction}"] = (
            cache_root / f"translation_d32_{direction}.pt"
        )
    output_root = artifacts_root / "representation/umap"
    output_names = (
        f"{backbone}_coordinates.csv",
        f"{backbone}_comparison.png",
        f"{backbone}_record.json",
    )
    if all((output_root / name).is_file() for name in output_names):
        prior = json.loads(
            (output_root / output_names[2]).read_text(encoding="utf-8")
        )
        if (
            prior.get("backbone") != backbone
            or prior.get("analysis_source_sha256") != sha256_file(Path(__file__))
            or prior.get("selection_manifest_sha256") != sha256_file(selection_path)
            or prior.get("software_versions", {}).get("umap_learn")
            != umap_version
            or prior.get("coordinate_csv_sha256")
            != sha256_file(output_root / output_names[0])
            or prior.get("figure_png_sha256")
            != sha256_file(output_root / output_names[1])
            or prior.get("feature_cache_sha256") != {
                str(path.relative_to(artifacts_root)): sha256_file(path)
                for path in input_paths.values()
            }
        ):
            raise RuntimeError(f"Existing UMAP output is invalid: {backbone}")
        for name in output_names:
            write_once(
                repo_root / "task1/results/representation/umap" / name,
                (output_root / name).read_bytes(),
            )
        print(f"PASS: reused verified {backbone} combined UMAP output.")
        return
    caches = {
        name: read_features(path, backbone, 200 if name == "cue_conflict" else 500,
                            width)
        for name, path in input_paths.items()
    }
    weights = caches["clean"]["metadata"].get("weights")
    representation = caches["clean"]["metadata"].get("representation")
    if any(
        cache["metadata"].get("weights") != weights
        or cache["metadata"].get("representation") != representation
        for cache in caches.values()
    ):
        raise RuntimeError("Feature caches use different frozen weights.")
    clean_indices = caches["clean"].get("official_indices")
    clean_labels = caches["clean"].get("labels")
    if (
        not isinstance(clean_indices, Tensor)
        or not isinstance(clean_labels, Tensor)
        or tuple(clean_indices.shape) != (500,)
        or tuple(clean_labels.shape) != (500,)
        or len(set(clean_indices.tolist())) != 500
    ):
        raise RuntimeError("Clean feature identities are invalid.")
    index_to_position = {
        index: position for position, index in enumerate(clean_indices.tolist())
    }
    selected_indices = [row["clean_official_index"] for row in selection_rows]
    selected_labels = [row["true_class_id"] for row in selection_rows]
    if any(index not in index_to_position for index in selected_indices):
        raise RuntimeError("Selected clean image is missing from cache.")
    positions = torch.tensor(
        [index_to_position[index] for index in selected_indices],
        dtype=torch.long,
    )
    if clean_labels[positions].tolist() != selected_labels:
        raise RuntimeError("Selected class labels differ from clean cache.")
    for condition, cache in caches.items():
        if condition == "cue_conflict":
            if (
                cache.get("candidate_ids")
                != [row["candidate_id"] for row in selection_rows]
                or not torch.equal(cache.get("content_indices"),
                                   torch.tensor(selected_indices))
                or not torch.equal(cache.get("shape_labels"),
                                   torch.tensor(selected_labels))
            ):
                raise RuntimeError("Cue feature order differs from selection.")
        elif (
            not torch.equal(cache.get("official_indices"), clean_indices)
            or not torch.equal(cache.get("labels"), clean_labels)
        ):
            raise RuntimeError(f"Feature order differs: {condition}")

    selected_features: dict[str, Tensor] = {
        "clean": caches["clean"]["features"][positions],
        "grayscale": caches["grayscale"]["features"][positions],
        "cue_conflict": caches["cue_conflict"]["features"],
        "patch_shuffle": caches["patch_shuffle"]["features"][positions],
    }
    translation = torch.empty((SIZE, width), dtype=torch.float32)
    for row_position, (row, clean_position) in enumerate(
        zip(selection_rows, positions.tolist())
    ):
        condition = f"translation_32px_{row['translation_direction']}"
        translation[row_position] = caches[condition]["features"][clean_position]
    selected_features["translation_32px"] = translation
    combined = torch.cat([selected_features[name] for name in CONDITIONS])
    if tuple(combined.shape) != (5 * SIZE, width):
        raise RuntimeError("Combined UMAP matrix has the wrong shape.")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=SEED,
        transform_seed=SEED,
    )
    coordinates = np.asarray(
        reducer.fit_transform(combined.numpy()), dtype=np.float32
    )
    if coordinates.shape != (5 * SIZE, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("UMAP returned invalid coordinates.")

    coordinate_rows: list[dict[str, Any]] = []
    for condition_position, condition in enumerate(CONDITIONS):
        for row_position, row in enumerate(selection_rows):
            x, y = coordinates[condition_position * SIZE + row_position]
            coordinate_rows.append({
                "backbone": backbone,
                "condition": condition,
                "condition_point_id": (
                    row["candidate_id"] if condition == "cue_conflict"
                    else row["clean_official_index"]
                ),
                "clean_official_index": row["clean_official_index"],
                "true_class_id": row["true_class_id"],
                "translation_direction": (
                    row["translation_direction"]
                    if condition == "translation_32px" else ""
                ),
                "umap_x": float(x),
                "umap_y": float(y),
            })
    coordinate_bytes = csv_bytes(coordinate_rows, COORDINATE_FIELDS)

    labels = np.asarray(selected_labels)
    class_colors = plt.get_cmap("tab10")(np.arange(10))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    comparison_conditions = CONDITIONS[1:]
    extent = coordinates.max(axis=0) - coordinates.min(axis=0)
    padding = np.maximum(extent * 0.05, 0.25)
    x_limits = (coordinates[:, 0].min() - padding[0],
                coordinates[:, 0].max() + padding[0])
    y_limits = (coordinates[:, 1].min() - padding[1],
                coordinates[:, 1].max() + padding[1])
    for ax, condition in zip(axes.flat, comparison_conditions):
        transformed_position = CONDITIONS.index(condition)
        clean_xy = coordinates[:SIZE]
        transformed_xy = coordinates[
            transformed_position * SIZE : (transformed_position + 1) * SIZE
        ]
        for class_id in range(10):
            mask = labels == class_id
            color = class_colors[class_id]
            ax.scatter(
                clean_xy[mask, 0], clean_xy[mask, 1],
                s=18, marker="o", facecolors=[color], edgecolors="none",
                alpha=0.35, rasterized=True,
            )
            ax.scatter(
                transformed_xy[mask, 0], transformed_xy[mask, 1],
                s=21, marker="x", c=[color], linewidths=0.8,
                alpha=0.8, rasterized=True,
            )
        ax.set_title("Clean vs " + condition.replace("_", " "))
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.12)
    class_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor=class_colors[class_id], markeredgecolor="none",
               label=CLASS_NAMES[class_id])
        for class_id in range(10)
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black",
               markersize=5, label="clean"),
        Line2D([0], [0], marker="x", linestyle="None", color="black",
               markersize=6, label="transformed"),
    ]
    fig.legend(
        handles=class_handles + marker_handles,
        loc="lower center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        f"{backbone}: one UMAP fit to all five conditions "
        "(200 images per condition)",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92,
                        bottom=0.13, hspace=0.25, wspace=0.18)
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", dpi=300)
    plt.close(fig)
    image_bytes = image_buffer.getvalue()

    input_hashes = {
        str(path.relative_to(artifacts_root)): sha256_file(path)
        for path in input_paths.values()
    }
    record = {
        "backbone": backbone,
        "frozen_weights": weights,
        "representation": representation,
        "number_of_fit_points": 1000,
        "points_per_condition": SIZE,
        "conditions_in_fit_order": list(CONDITIONS),
        "one_combined_fit_per_backbone": True,
        "shared_axes_across_condition_panels": True,
        "selection_manifest_sha256": sha256_file(selection_path),
        "feature_cache_sha256": dict(sorted(input_hashes.items())),
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "umap_parameters": {
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "cosine",
            "random_state": SEED,
            "transform_seed": SEED,
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "umap_learn": umap_version,
            "matplotlib": matplotlib.__version__,
        },
        "coordinate_csv_sha256": hashlib.sha256(coordinate_bytes).hexdigest(),
        "figure_png_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "interpretation_rule": (
            "Compare clean and transformed conditions within a backbone; "
            "do not compare absolute coordinates across separately fitted backbones."
        ),
    }
    record_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    for root in (
        artifacts_root / "representation/umap",
        repo_root / "task1/results/representation/umap",
    ):
        write_once(root / f"{backbone}_coordinates.csv", coordinate_bytes)
        write_once(root / f"{backbone}_comparison.png", image_bytes)
        write_once(root / f"{backbone}_record.json", record_bytes)
    print(f"PASS: {backbone} combined 1,000-point UMAP and figure saved.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "plot"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=tuple(BACKBONES))
    args = parser.parse_args()
    if args.mode == "plot" and args.backbone is None:
        raise ValueError("--backbone is required for plotting.")
    rows, path, _ = build_selection(
        args.repo_root.resolve(), args.artifacts_root.resolve()
    )
    if args.mode == "plot":
        plot_backbone(
            args.backbone, args.repo_root.resolve(),
            args.artifacts_root.resolve(), rows, path,
        )


if __name__ == "__main__":
    main()
