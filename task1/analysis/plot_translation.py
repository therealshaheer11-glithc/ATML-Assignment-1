"""Audit saved Task 1 translation results and plot both required curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREDICTORS = (
    "resnet50_linear_head",
    "vit_b_16_linear_head",
    "clip_vit_b_32_linear_head",
    "clip_vit_b_32_zero_shot",
)
LABELS = (
    "ResNet-50 head",
    "ViT-B/16 head",
    "CLIP head",
    "CLIP zero-shot",
)
COLORS = ("#2166ac", "#b2182b", "#1b7837", "#762a83")
DIRECTIONS = ("up", "down", "left", "right")
DISPLACEMENTS = (0, 8, 16, 32)
FIELDS = (
    "predictor", "displacement_pixels", "absolute_accuracy",
    "accuracy_change_from_own_clean_baseline",
    "prediction_consistency_with_clean", "aggregation",
    "examples_per_direction", "number_of_directions",
)
TOLERANCE = 1e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing result differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary result: {temporary}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > TOLERANCE:
        raise RuntimeError(f"Inconsistent {label}: {actual} versus {expected}")


def collect(repo_root: Path, artifacts_root: Path) -> tuple[list[dict], dict]:
    clean_path = artifacts_root / "metrics/clean/clean_metrics.json"
    clean = json.loads(clean_path.read_text(encoding="utf-8"))["results"]
    rows: list[dict] = []
    source_hashes = {str(clean_path.relative_to(artifacts_root)): sha256_file(clean_path)}
    shared_input_hashes: dict[tuple[int, str], str] = {}

    for predictor in PREDICTORS:
        clean_accuracy = float(clean[predictor]["top1_accuracy"])
        if not 0 <= clean_accuracy <= 1:
            raise RuntimeError(f"Invalid clean accuracy: {predictor}")
        rows.append({
            "predictor": predictor,
            "displacement_pixels": 0,
            "absolute_accuracy": clean_accuracy,
            "accuracy_change_from_own_clean_baseline": 0.0,
            "prediction_consistency_with_clean": 1.0,
            "aggregation": "clean identity baseline",
            "examples_per_direction": 500,
            "number_of_directions": 1,
        })

        for displacement in DISPLACEMENTS[1:]:
            filename = f"{predictor}__d{displacement}_summary.json"
            path = artifacts_root / "metrics/translation" / filename
            summary = json.loads(path.read_text(encoding="utf-8"))
            repo_copy = repo_root / "task1/results/translation" / filename
            # A fresh Colab checkout may lack result copies that persist on Drive.
            # Restore missing copies, but never replace a different existing file.
            write_once(repo_copy, path.read_bytes())
            source_hashes[str(path.relative_to(artifacts_root))] = sha256_file(path)

            if (
                summary.get("predictor") != predictor
                or summary.get("displacement_pixels") != displacement
                or summary.get("directions") != list(DIRECTIONS)
                or summary.get("examples_per_direction") != 500
                or summary.get("aggregation") != "unweighted mean of four directions"
            ):
                raise RuntimeError(f"Incomplete translation summary: {filename}")
            close(float(summary["own_clean_accuracy"]), clean_accuracy,
                  f"clean accuracy in {filename}")
            details = summary.get("direction_results")
            if not isinstance(details, dict) or set(details) != set(DIRECTIONS):
                raise RuntimeError(f"Missing directions: {filename}")

            for direction in DIRECTIONS:
                detail = details[direction]
                if (
                    detail.get("predictor") != predictor
                    or detail.get("displacement_pixels") != displacement
                    or detail.get("direction") != direction
                    or detail.get("number_of_examples") != 500
                ):
                    raise RuntimeError(f"Invalid {direction} result: {filename}")
                close(float(detail["absolute_accuracy"]),
                      int(detail["correct_count"]) / 500,
                      f"{direction} accuracy in {filename}")
                close(float(detail["prediction_consistency_with_clean"]),
                      int(detail["prediction_match_count"]) / 500,
                      f"{direction} consistency in {filename}")
                input_hash = detail["input_manifest_sha256"]
                key = (displacement, direction)
                if key in shared_input_hashes and shared_input_hashes[key] != input_hash:
                    raise RuntimeError(f"Models received different images: {key}")
                shared_input_hashes[key] = input_hash

            accuracy = sum(float(details[d]["absolute_accuracy"]) for d in DIRECTIONS) / 4
            consistency = sum(
                float(details[d]["prediction_consistency_with_clean"])
                for d in DIRECTIONS
            ) / 4
            close(float(summary["mean_accuracy"]), accuracy,
                  f"mean accuracy in {filename}")
            close(float(summary["mean_prediction_consistency_with_clean"]),
                  consistency, f"mean consistency in {filename}")
            close(float(summary["mean_accuracy_change_from_own_clean_baseline"]),
                  accuracy - clean_accuracy, f"accuracy delta in {filename}")
            if not 0 <= accuracy <= 1 or not 0 <= consistency <= 1:
                raise RuntimeError(f"Out-of-range metric in {filename}")
            rows.append({
                "predictor": predictor,
                "displacement_pixels": displacement,
                "absolute_accuracy": accuracy,
                "accuracy_change_from_own_clean_baseline": accuracy - clean_accuracy,
                "prediction_consistency_with_clean": consistency,
                "aggregation": "unweighted mean of four directions",
                "examples_per_direction": 500,
                "number_of_directions": 4,
            })

    if len(rows) != 16 or len(shared_input_hashes) != 12:
        raise RuntimeError("Translation curve has missing evidence.")
    record = {
        "displacements_pixels": list(DISPLACEMENTS),
        "predictors": list(PREDICTORS),
        "zero_displacement": "The clean identity baseline: accuracy is clean top-1; consistency is 1.",
        "nonzero_aggregation": "Unweighted mean of all four cardinal directions, 500 images each.",
        "same_transformed_images_across_predictors": True,
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "shared_translation_input_manifest_sha256": {
            f"d{displacement}_{direction}": digest
            for (displacement, direction), digest in sorted(shared_input_hashes.items())
        },
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
    }
    return rows, record


def render(rows: list[dict]) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    metrics = (
        ("absolute_accuracy", "Top-1 accuracy"),
        ("prediction_consistency_with_clean", "Prediction consistency with clean"),
    )
    for ax, (field, title) in zip(axes, metrics):
        values = []
        for predictor, label, color in zip(PREDICTORS, LABELS, COLORS):
            predictor_rows = [row for row in rows if row["predictor"] == predictor]
            y = [float(row[field]) * 100 for row in predictor_rows]
            values.extend(y)
            ax.plot(DISPLACEMENTS, y, marker="o", markersize=5,
                    linewidth=2, color=color, label=label)
        ax.set_title(title)
        ax.set_xticks(DISPLACEMENTS)
        ax.set_xlabel("Translation displacement (pixels)")
        ax.set_ylabel("Percent (%)")
        ax.grid(alpha=0.2)
        ax.set_ylim(max(0, min(values) - 2), min(100, max(values) + 2))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("STL-10 translation: 500 images, four-direction mean", fontsize=14)
    fig.text(0.5, 0.035, "Y-axes are zoomed; exact values are in translation_curves.csv.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82,
                        bottom=0.23, wspace=0.24)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300)
    plt.close(fig)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    rows, record = collect(repo_root, artifacts_root)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    csv_data = buffer.getvalue().encode("utf-8")
    png_data = render(rows)
    record["curve_csv_sha256"] = hashlib.sha256(csv_data).hexdigest()
    record["curve_png_sha256"] = hashlib.sha256(png_data).hexdigest()
    record_data = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    for root in (
        artifacts_root / "figures/translation",
        repo_root / "task1/results/translation/figures",
    ):
        write_once(root / "translation_curves.csv", csv_data)
        write_once(root / "translation_curves.png", png_data)
        write_once(root / "translation_curves_record.json", record_data)

    for row in rows:
        print(
            f"{row['predictor']} d{row['displacement_pixels']}: "
            f"accuracy={row['absolute_accuracy']:.4f}, "
            f"consistency={row['prediction_consistency_with_clean']:.4f}"
        )
    print("PASS: all 16 translation points and shared inputs verified; figure saved.")


if __name__ == "__main__":
    main()
