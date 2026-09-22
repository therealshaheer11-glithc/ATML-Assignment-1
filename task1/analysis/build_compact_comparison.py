"""Build the Task 1 comparison from saved predictions; no model execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREDICTORS = {
    "resnet50_linear_head": "ResNet-50 head",
    "vit_b_16_linear_head": "ViT-B/16 head",
    "clip_vit_b_32_linear_head": "CLIP head",
    "clip_vit_b_32_zero_shot": "CLIP zero-shot",
}
CONDITIONS = ("clean", "grayscale", "hue_rotation_90", "patch_shuffle")
FIELDS = (
    "predictor", "condition", "number_of_examples", "correct_count",
    "accuracy_percent", "accuracy_change_pp", "prediction_match_count",
    "prediction_consistency_percent", "macro_f1_percent",
    "mean_maximum_confidence_percent",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, label: str) -> None:
    require(math.isfinite(float(actual)) and abs(float(actual) - expected) < 1e-7,
            f"Inconsistent {label}: {actual} versus {expected}")


def collect(artifacts: Path, repo: Path) -> tuple[list[dict], dict]:
    hashes = {}

    def read(relative: str, kind: str):
        data = (artifacts / relative).read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        return (json.loads(data) if kind == "json" else
                list(csv.DictReader(io.StringIO(data.decode("utf-8")))))

    clean = read("metrics/clean/clean_metrics.json", "json")
    color = read("metrics/color/color_metrics.json", "json")
    clean_predictions = read("metrics/clean/clean_predictions.csv", "csv")
    color_predictions = read("metrics/color/color_predictions.csv", "csv")
    manifest_data = (repo / "task1/results/splits/evaluation_subset.csv").read_bytes()
    manifest_hash = hashlib.sha256(manifest_data).hexdigest()
    manifest = list(csv.DictReader(io.StringIO(manifest_data.decode("utf-8"))))
    ids = [x["official_index"] for x in manifest]
    labels = {x["official_index"]: int(x["class_id"]) for x in manifest}
    require(len(ids) == len(set(ids)) == 500, "Expected 500 unique test images.")
    require(clean["manifest_sha256"] == color["manifest_sha256"] == manifest_hash,
            "Clean/color results refer to a different test subset.")
    require(set(clean["results"]) == set(PREDICTORS), "Unexpected predictors.")
    output = []

    for predictor in PREDICTORS:
        baseline = clean["results"][predictor]
        clean_rows = [x for x in clean_predictions if x["predictor"] == predictor]
        require([x["official_index"] for x in clean_rows] == ids,
                f"Clean image order differs: {predictor}")
        clean_by_id = {x["official_index"]: int(x["predicted_class_id"]) for x in clean_rows}
        clean_correct = sum(clean_by_id[i] == labels[i] for i in ids)
        f1_values = []
        for k in range(10):
            tp = sum(labels[i] == clean_by_id[i] == k for i in ids)
            denominator = sum(labels[i] == k for i in ids) + sum(clean_by_id[i] == k for i in ids)
            f1_values.append(2 * tp / denominator if denominator else 0.0)
        close(baseline["macro_f1"], fmean(f1_values), f"{predictor} macro-F1")
        close(baseline["mean_maximum_confidence"],
              fmean(float(x["maximum_confidence"]) for x in clean_rows),
              f"{predictor} clean confidence")

        patch = read(f"metrics/patch_shuffle/{predictor}.json", "json")
        patch_rows = read(f"metrics/patch_shuffle/{predictor}_predictions.csv", "csv")
        require(patch["evaluation_manifest_sha256"] == manifest_hash,
                f"Patch results use a different subset: {predictor}")
        for condition in CONDITIONS:
            if condition == "clean":
                data, metric, pred_key = clean_rows, baseline, "predicted_class_id"
            elif condition == "patch_shuffle":
                data, metric, pred_key = patch_rows, patch, "shuffled_predicted_class_id"
            else:
                data = [x for x in color_predictions
                        if x["predictor"] == predictor and x["condition"] == condition]
                metric = color["results"][predictor][condition]
                pred_key = "transformed_predicted_class_id"
            require([x["official_index"] for x in data] == ids,
                    f"Image order differs: {predictor}, {condition}")
            correct = matches = 0
            for x in data:
                index, predicted = x["official_index"], int(x[pred_key])
                require(int(x["true_class_id"]) == labels[index], "Ground-truth label mismatch.")
                if condition != "clean":
                    require(int(x["clean_predicted_class_id"]) == clean_by_id[index],
                            "Saved clean prediction mismatch.")
                correct += predicted == labels[index]
                matches += predicted == clean_by_id[index]
            require(metric["number_of_examples"] == 500 and metric["correct_count"] == correct,
                    f"Count mismatch: {predictor}, {condition}")
            accuracy_key = "top1_accuracy" if condition == "clean" else "absolute_accuracy"
            close(metric[accuracy_key], correct / 500, "accuracy")
            if condition != "clean":
                require(metric["prediction_match_count"] == matches, "Prediction match count mismatch.")
                close(metric["prediction_consistency_with_clean"], matches / 500, "consistency")
                close(metric["accuracy_change_from_own_clean_baseline"],
                      (correct - clean_correct) / 500, "accuracy change")
            output.append({
                "predictor": predictor,
                "condition": condition,
                "number_of_examples": 500,
                "correct_count": correct,
                "accuracy_percent": 100 * correct / 500,
                "accuracy_change_pp": 100 * (correct - clean_correct) / 500,
                "prediction_match_count": matches,
                "prediction_consistency_percent": 100 * matches / 500,
                "macro_f1_percent": 100 * baseline["macro_f1"] if condition == "clean" else "",
                "mean_maximum_confidence_percent": (
                    100 * baseline["mean_maximum_confidence"] if condition == "clean" else ""
                ),
            })
    require(len(output) == 16, "Expected four predictors by four conditions.")
    return output, {
        "source_file_sha256": dict(sorted(hashes.items())),
        "evaluation_manifest_sha256": manifest_hash,
        "accuracy_definition": "100 * correct_count / number_of_examples",
        "accuracy_change_definition": "transformed accuracy minus own clean accuracy, in percentage points",
        "consistency_definition": "100 * prediction_match_count / number_of_examples",
        "clean_f1_and_confidence": "Saved clean macro-F1 and mean maximum confidence, multiplied by 100",
        "missing_csv_values": "Clean-only F1 and confidence fields are blank for transformed conditions",
        "figure_decimal_places": 1,
        "models_executed": False,
        "report_interpretation_generated": False,
    }


def render(rows: list[dict]) -> bytes:
    lookup = {(x["predictor"], x["condition"]): x for x in rows}
    cells = []
    for predictor, label in PREDICTORS.items():
        clean = lookup[predictor, "clean"]
        line = [label, f"{clean['accuracy_percent']:.1f} / {clean['macro_f1_percent']:.1f} / "
                       f"{clean['mean_maximum_confidence_percent']:.1f}"]
        for condition in CONDITIONS[1:]:
            value = lookup[predictor, condition]
            line.append(f"{value['accuracy_percent']:.1f} / {value['accuracy_change_pp']:+.1f} / "
                        f"{value['prediction_consistency_percent']:.1f}")
        cells.append(line)
    fig, ax = plt.subplots(figsize=(8.5, 2.1))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=["Predictor", "Clean\nAcc / F1 / Conf", "Grayscale\nAcc / Δ / Cons",
                   "Hue +90°\nAcc / Δ / Cons", "Patch shuffle\nAcc / Δ / Cons"],
        cellLoc="center", colLoc="center", colWidths=[0.18, 0.205, 0.205, 0.205, 0.205],
        bbox=[0.005, 0.22, 0.99, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#ccd2d9")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(weight="bold", fontsize=8.8)
        elif row % 2 == 0:
            cell.set_facecolor("#f7f9fb")
        if column == 0 and row:
            cell.set_text_props(ha="left")
            cell.PAD = 0.07
    fig.text(0.5, 0.135,
             "Acc: accuracy (%)    F1: macro-F1 (×100)    Conf: mean maximum confidence (%)",
             ha="center", fontsize=8)
    fig.text(0.5, 0.045,
             "Δ: accuracy change from clean (percentage points)    Cons: prediction consistency (%)",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.08)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300)
    plt.close(fig)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    args = parser.parse_args()
    repo, artifacts = args.repo_root.resolve(), args.artifacts_root.resolve()
    rows, record = collect(artifacts, repo)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    csv_data = buffer.getvalue().encode("utf-8")
    png_data = render(rows)
    record.update({
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_data).hexdigest(),
        "figure_sha256": hashlib.sha256(png_data).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
    })
    outputs = {
        "compact_comparison.csv": csv_data,
        "compact_comparison.png": png_data,
        "compact_comparison_record.json": (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
    }
    pending = []
    for root in (artifacts / "figures/comparison", repo / "task1/results/comparison"):
        for name, data in outputs.items():
            path = root / name
            if path.exists():
                require(path.read_bytes() == data, f"Existing output differs; no files replaced: {path}")
            else:
                pending.append((path, data))
    for path, data in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(data)
    print("PASS: 16 comparison rows verified against saved predictions.")
    print("PASS: compact table PNG, CSV, and provenance saved to Drive and repository.")
    print("No model execution; no report interpretation generated.")


if __name__ == "__main__":
    main()
