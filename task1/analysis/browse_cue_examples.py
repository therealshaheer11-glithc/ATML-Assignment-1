"""Browse all frozen AdaIN conflicts alongside four saved model predictions.

This is a read-only review aid. It does not select examples for the report.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image
from torchvision.datasets import STL10


PREDICTORS = (
    ("resnet50_linear_head", "ResNet-50 head"),
    ("vit_b_16_linear_head", "ViT-B/16 head"),
    ("clip_vit_b_32_linear_head", "CLIP head"),
    ("clip_vit_b_32_zero_shot", "CLIP zero-shot"),
)
CLASS_NAMES = (
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch(artifacts_root: str | Path) -> None:
    root = Path(artifacts_root).resolve()
    accepted_path = root / "cue_conflict/manifests/accepted_conflicts.csv"
    selection_record = json.loads(
        (root / "cue_conflict/manifests/selection_record.json").read_text()
    )
    if selection_record["accepted_conflicts_sha256"] != sha256(accepted_path):
        raise RuntimeError("Frozen accepted manifest changed.")
    accepted = read_csv(accepted_path)
    ids = [row["candidate_id"] for row in accepted]
    if len(ids) != 200 or len(set(ids)) != 200:
        raise RuntimeError("Expected 200 unique accepted cue conflicts.")
    if Counter(row["direction"] for row in accepted) != {
        direction: 20 for direction in {row["direction"] for row in accepted}
    } or len({row["direction"] for row in accepted}) != 10:
        raise RuntimeError("Accepted directions are not balanced.")
    by_id = {row["candidate_id"]: row for row in accepted}

    predictions: dict[str, dict[str, dict[str, str]]] = {}
    for predictor, _ in PREDICTORS:
        path = root / "metrics/cue_conflict" / f"{predictor}_predictions.csv"
        rows = read_csv(path)
        if [row["candidate_id"] for row in rows] != ids:
            raise RuntimeError(f"Prediction order differs: {predictor}")
        for row in rows:
            source = by_id[row["candidate_id"]]
            predicted_class_id = int(row["predicted_class_id"])
            shape_id = int(source["shape_class_id"])
            texture_id = int(source["texture_class_id"])
            expected_decision = (
                "shape" if predicted_class_id == shape_id else
                "texture" if predicted_class_id == texture_id else "other"
            )
            if (
                row["output_sha256"] != source["output_sha256"]
                or int(row["shape_class_id"]) != shape_id
                or int(row["texture_class_id"]) != texture_id
                or row["predicted_class_name"] != CLASS_NAMES[predicted_class_id]
                or row["decision"] != expected_decision
            ):
                raise RuntimeError(f"Prediction evidence differs: {predictor}")
        predictions[predictor] = {row["candidate_id"]: row for row in rows}

    test_data = STL10(
        root=str(root / "datasets/stl10"), split="test", download=False
    )
    if len(test_data) != 8000:
        raise RuntimeError("Official STL-10 test images are unavailable.")

    def decisions(candidate_id: str) -> list[str]:
        return [predictions[name][candidate_id]["decision"] for name, _ in PREDICTORS]

    filters = {
        "All 200": ids,
        "All four predict shape": [
            candidate_id for candidate_id in ids
            if all(decision == "shape" for decision in decisions(candidate_id))
        ],
        "At least one texture decision": [
            candidate_id for candidate_id in ids
            if "texture" in decisions(candidate_id)
        ],
        "At least one other decision": [
            candidate_id for candidate_id in ids
            if "other" in decisions(candidate_id)
        ],
        "Models predict different classes": [
            candidate_id for candidate_id in ids
            if len({predictions[name][candidate_id]["predicted_class_id"]
                    for name, _ in PREDICTORS}) > 1
        ],
        "CLIP head and zero-shot disagree": [
            candidate_id for candidate_id in ids
            if predictions["clip_vit_b_32_linear_head"][candidate_id]["predicted_class_id"]
            != predictions["clip_vit_b_32_zero_shot"][candidate_id]["predicted_class_id"]
        ],
    }
    print("Read-only browser: no report examples have been selected.")
    print("Filter counts:")
    for label, values in filters.items():
        print(f"  {label}: {len(values)}")

    filter_widget = widgets.Dropdown(options=list(filters), description="Show:",
                                     layout=widgets.Layout(width="420px"))
    candidate_widget = widgets.Dropdown(description="Candidate:",
                                        layout=widgets.Layout(width="420px"))
    output = widgets.Output()

    def update_candidates(change: dict | None = None) -> None:
        candidate_widget.options = filters[filter_widget.value]
        if not candidate_widget.options:
            with output:
                clear_output(wait=True)
                print("No images in this category.")

    def render(change: dict | None = None) -> None:
        candidate_id = candidate_widget.value
        if not candidate_id:
            return
        row = by_id[candidate_id]
        output_path = root / "cue_conflict/candidates" / f"{candidate_id}.png"
        if sha256(output_path) != row["output_sha256"]:
            raise RuntimeError(f"Generated image changed: {candidate_id}")
        content_index = int(row["content_official_index"])
        style_index = int(row["style_official_index"])
        content_image, content_label = test_data[content_index]
        style_image, style_label = test_data[style_index]
        if (
            content_label != int(row["shape_class_id"])
            or style_label != int(row["texture_class_id"])
        ):
            raise RuntimeError(f"STL-10 source labels changed: {candidate_id}")
        with Image.open(output_path) as image:
            output_image = image.convert("RGB").copy()
        if output_image.size != (224, 224):
            raise RuntimeError(f"Generated image size changed: {candidate_id}")
        with output:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, image, title in zip(
                axes,
                (content_image, output_image, style_image),
                (f"Content / shape: {CLASS_NAMES[content_label]}",
                 f"AdaIN output: {candidate_id}",
                 f"Style / texture: {CLASS_NAMES[style_label]}"),
            ):
                ax.imshow(image)
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            for predictor, label in PREDICTORS:
                prediction = predictions[predictor][candidate_id]
                print(
                    f"{label:17s}  "
                    f"{prediction['predicted_class_name']:9s}  "
                    f"{prediction['decision']:7s}  "
                    f"confidence={float(prediction['maximum_confidence']):.3f}"
                )

    filter_widget.observe(update_candidates, names="value")
    candidate_widget.observe(render, names="value")
    display(widgets.VBox([filter_widget, candidate_widget, output]))
    update_candidates()
    render()
