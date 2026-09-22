"""Correct translation-figure spacing using the frozen curve CSV only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREDICTORS = (
    "resnet50_linear_head", "vit_b_16_linear_head",
    "clip_vit_b_32_linear_head", "clip_vit_b_32_zero_shot",
)
LABELS = ("ResNet-50 head", "ViT-B/16 head", "CLIP head", "CLIP zero-shot")
COLORS = ("#2166ac", "#b2182b", "#1b7837", "#762a83")
DISPLACEMENTS = (0, 8, 16, 32)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_curves(repo: Path, artifacts: Path) -> tuple[list[dict], dict]:
    directory = artifacts / "figures/translation"
    csv_path = directory / "translation_curves.csv"
    record_path = directory / "translation_curves_record.json"
    original = json.loads(record_path.read_text(encoding="utf-8"))
    require(sha(csv_path) == original["curve_csv_sha256"], "Frozen curve CSV changed.")
    require(sha(directory / "translation_curves.png") == original["curve_png_sha256"],
            "Original figure changed.")
    require(sha(repo / "task1/analysis/plot_translation.py") == original["analysis_source_sha256"],
            "Original plotting script differs from its recorded version.")
    for relative, digest in original["source_file_sha256"].items():
        path = (artifacts / relative).resolve()
        require(path.is_relative_to(artifacts), "Invalid recorded input path.")
        require(sha(path) == digest, f"Underlying result changed: {relative}")
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    expected = {(p, d) for p in PREDICTORS for d in DISPLACEMENTS}
    keys = [(x["predictor"], int(x["displacement_pixels"])) for x in rows]
    require(len(rows) == 16 and set(keys) == expected, "Missing or duplicate curve points.")
    for row in rows:
        for field in ("absolute_accuracy", "prediction_consistency_with_clean"):
            value = float(row[field])
            require(math.isfinite(value) and 0 <= value <= 1, f"Invalid value: {field}")
        require(int(row["examples_per_direction"]) == 500, "Unexpected sample count.")
        displacement = int(row["displacement_pixels"])
        if displacement:
            require(int(row["number_of_directions"]) == 4 and
                    row["aggregation"] == "unweighted mean of four directions",
                    "Unexpected direction aggregation.")
        else:
            require(float(row["prediction_consistency_with_clean"]) == 1.0,
                    "Invalid identity baseline.")
    return rows, {
        "source_csv": "figures/translation/translation_curves.csv",
        "source_csv_sha256": sha(csv_path),
        "original_record_sha256": sha(record_path),
        "original_figure_sha256": original["curve_png_sha256"],
        "original_plotting_source_sha256": original["analysis_source_sha256"],
        "source_result_sha256": original["source_file_sha256"],
        "number_of_curve_points": 16,
        "change": "Increase figure height and separate axis labels, legend, and footer",
        "curve_values_changed": False,
        "axis_limits_rule_changed": False,
        "model_inference_performed": False,
    }


def render(rows: list[dict]) -> tuple[bytes, dict]:
    lookup = {(x["predictor"], int(x["displacement_pixels"])): x for x in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    metrics = (
        ("absolute_accuracy", "Top-1 accuracy"),
        ("prediction_consistency_with_clean", "Prediction consistency with clean"),
    )
    axis_limits = {}
    for ax, (field, title) in zip(axes, metrics):
        values = []
        for predictor, label, color in zip(PREDICTORS, LABELS, COLORS):
            y = [float(lookup[predictor, d][field]) * 100 for d in DISPLACEMENTS]
            values.extend(y)
            ax.plot(DISPLACEMENTS, y, marker="o", markersize=5,
                    linewidth=2, color=color, label=label)
        ax.set_title(title)
        ax.set_xticks(DISPLACEMENTS)
        ax.set_xlabel("Translation displacement (pixels)")
        ax.set_ylabel("Percent (%)")
        ax.grid(alpha=0.2)
        limits = (max(0, min(values) - 2), min(100, max(values) + 2))
        ax.set_ylim(*limits)
        axis_limits[field] = list(limits)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="center", ncol=4,
                        frameon=False, bbox_to_anchor=(0.5, 0.12))
    fig.suptitle("STL-10 translation: 500 images, four-direction mean", fontsize=14)
    footer = fig.text(
        0.5, 0.035, "Y-axes are zoomed; exact values are in translation_curves.csv.",
        ha="center", va="bottom", fontsize=8,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83,
                        bottom=0.30, wspace=0.24)
    # Check the actual text extents before exporting the final layout.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    footer_box = footer.get_window_extent(renderer)
    require(not legend_box.overlaps(footer_box), "Legend overlaps footer.")
    for ax in axes:
        require(not ax.xaxis.label.get_window_extent(renderer).overlaps(legend_box),
                "Axis label overlaps legend.")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300)
    plt.close(fig)
    return buffer.getvalue(), axis_limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    args = parser.parse_args()
    repo, artifacts = args.repo_root.resolve(), args.artifacts_root.resolve()
    rows, record = read_curves(repo, artifacts)
    png, limits = render(rows)
    record.update({
        "final_figure_sha256": hashlib.sha256(png).hexdigest(),
        "analysis_source_sha256": sha(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "axis_limits_percent": limits,
        "legend_and_footer_overlap": False,
    })
    outputs = {
        "translation_curves_final.png": png,
        "translation_curves_final_record.json":
            (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
    }
    pending = []
    for directory in (artifacts / "figures/translation",
                      repo / "task1/results/translation/figures"):
        for name, data in outputs.items():
            path = directory / name
            if path.exists():
                require(path.read_bytes() == data, f"Existing output differs: {path}")
            else:
                pending.append((path, data))
    for path, data in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(data)
    print("PASS: all 16 saved curve points and source hashes verified.")
    print("PASS: final translation figure saved with separate legend and footer.")
    print("Original figure and numeric results preserved; no model execution.")


if __name__ == "__main__":
    main()
