"""Export three user-approved cue-conflict examples and saved predictions."""

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
from PIL import Image
from torchvision.datasets import STL10


SELECTED_IDS = (
    "p02_r01_a_to_b",
    "p01_r19_a_to_b",
    "p02_r09_a_to_b",
)
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
CSV_FIELDS = (
    "candidate_id", "content_official_index", "style_official_index",
    "shape_class_id", "shape_class_name", "texture_class_id",
    "texture_class_name", "predictor", "predicted_class_id",
    "predicted_class_name", "decision", "maximum_confidence",
    "output_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing output differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Unresolved temporary output: {temporary}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def collect(artifacts_root: Path) -> tuple[list[dict[str, str]], dict]:
    manifest = artifacts_root / "cue_conflict/manifests/accepted_conflicts.csv"
    selection_record_path = (
        artifacts_root / "cue_conflict/manifests/selection_record.json"
    )
    selection_record = json.loads(
        selection_record_path.read_text(encoding="utf-8")
    )
    if selection_record.get("accepted_conflicts_sha256") != sha256_file(manifest):
        raise RuntimeError("Accepted cue-conflict manifest is no longer frozen.")
    accepted = read_csv(manifest)
    by_id = {row["candidate_id"]: row for row in accepted}
    if len(accepted) != 200 or len(by_id) != 200:
        raise RuntimeError("Expected 200 unique accepted conflicts.")
    if any(candidate_id not in by_id for candidate_id in SELECTED_IDS):
        raise RuntimeError("A selected candidate is not in the accepted set.")

    prediction_paths: dict[str, Path] = {}
    prediction_maps: dict[str, dict[str, dict[str, str]]] = {}
    accepted_order = [row["candidate_id"] for row in accepted]
    for predictor, _ in PREDICTORS:
        path = (
            artifacts_root / "metrics/cue_conflict"
            / f"{predictor}_predictions.csv"
        )
        prediction_paths[predictor] = path
        predictor_rows = read_csv(path)
        if [row["candidate_id"] for row in predictor_rows] != accepted_order:
            raise RuntimeError(f"Prediction order differs: {predictor}")
        prediction_maps[predictor] = {
            row["candidate_id"]: row for row in predictor_rows
        }

    selected_rows: list[dict[str, str]] = []
    image_hashes: dict[str, str] = {}
    for candidate_id in SELECTED_IDS:
        source = by_id[candidate_id]
        content_id = int(source["content_class_id"])
        style_id = int(source["style_class_id"])
        if (
            int(source["shape_class_id"]) != content_id
            or int(source["texture_class_id"]) != style_id
            or content_id == style_id
            or source["review_status"] != "accepted"
            or source["review_completed_before_model_evaluation"] != "True"
        ):
            raise RuntimeError(f"Invalid accepted cue row: {candidate_id}")
        output_path = (
            artifacts_root / "cue_conflict/candidates"
            / f"{candidate_id}.png"
        )
        image_hash = sha256_file(output_path)
        if image_hash != source["output_sha256"]:
            raise RuntimeError(f"Candidate image changed: {candidate_id}")
        with Image.open(output_path) as output_image:
            if output_image.mode != "RGB" or output_image.size != (224, 224):
                raise RuntimeError(f"Invalid candidate image: {candidate_id}")
        image_hashes[candidate_id] = image_hash

        for predictor, _ in PREDICTORS:
            prediction = prediction_maps[predictor][candidate_id]
            predicted_id = int(prediction["predicted_class_id"])
            decision = (
                "shape" if predicted_id == content_id else
                "texture" if predicted_id == style_id else "other"
            )
            confidence = float(prediction["maximum_confidence"])
            if (
                prediction["output_sha256"] != image_hash
                or int(prediction["shape_class_id"]) != content_id
                or int(prediction["texture_class_id"]) != style_id
                or prediction["predicted_class_name"] != CLASS_NAMES[predicted_id]
                or prediction["decision"] != decision
                or not 0 <= confidence <= 1
            ):
                raise RuntimeError(f"Prediction mismatch: {candidate_id}, {predictor}")
            selected_rows.append({
                "candidate_id": candidate_id,
                "content_official_index": source["content_official_index"],
                "style_official_index": source["style_official_index"],
                "shape_class_id": content_id,
                "shape_class_name": CLASS_NAMES[content_id],
                "texture_class_id": style_id,
                "texture_class_name": CLASS_NAMES[style_id],
                "predictor": predictor,
                "predicted_class_id": predicted_id,
                "predicted_class_name": CLASS_NAMES[predicted_id],
                "decision": decision,
                "maximum_confidence": confidence,
                "output_sha256": image_hash,
            })

    if len(selected_rows) != 12:
        raise RuntimeError("Expected 3 examples by 4 predictors.")
    metadata = {
        "selected_candidate_ids_in_figure_order": list(SELECTED_IDS),
        "selection_confirmed_by_user": True,
        "accepted_manifest_sha256": sha256_file(manifest),
        "selection_record_sha256": sha256_file(selection_record_path),
        "prediction_csv_sha256": {
            predictor: sha256_file(path)
            for predictor, path in prediction_paths.items()
        },
        "candidate_png_sha256": image_hashes,
        "model_inference_rerun": False,
        "report_interpretation_or_caption_generated": False,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
    }
    return selected_rows, metadata


def render(artifacts_root: Path, selected_rows: list[dict[str, str]]) -> bytes:
    dataset = STL10(
        root=str(artifacts_root / "datasets/stl10"),
        split="test", download=False,
    )
    if len(dataset) != 8000:
        raise RuntimeError("Official STL-10 test partition is missing.")
    fig, axes = plt.subplots(
        3, 4, figsize=(14, 10.5),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.15]},
    )
    for row_number, candidate_id in enumerate(SELECTED_IDS):
        predictions = [
            row for row in selected_rows if row["candidate_id"] == candidate_id
        ]
        reference = predictions[0]
        content_image, content_label = dataset[
            int(reference["content_official_index"])
        ]
        style_image, style_label = dataset[
            int(reference["style_official_index"])
        ]
        if (
            content_label != reference["shape_class_id"]
            or style_label != reference["texture_class_id"]
        ):
            raise RuntimeError(f"Official STL-10 label mismatch: {candidate_id}")
        output_path = (
            artifacts_root / "cue_conflict/candidates"
            / f"{candidate_id}.png"
        )
        with Image.open(output_path) as image:
            stylized_image = image.copy()
        images = (content_image, stylized_image, style_image)
        labels = (
            f"Content: {reference['shape_class_name']}",
            f"AdaIN: {candidate_id}",
            f"Style: {reference['texture_class_name']}",
        )
        for column, (image, label) in enumerate(zip(images, labels)):
            ax = axes[row_number, column]
            ax.imshow(image)
            ax.set_title(label, fontsize=10)
            ax.axis("off")
        ax = axes[row_number, 3]
        ax.axis("off")
        ax.set_title("Saved predictions", fontsize=10)
        lines = []
        for prediction, (_, display_name) in zip(predictions, PREDICTORS):
            lines.append(
                f"{display_name}\n"
                f"  {prediction['predicted_class_name']} "
                f"({prediction['decision']})"
            )
        ax.text(
            0.03, 0.95, "\n\n".join(lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, linespacing=1.2,
        )
    fig.suptitle("Cue-conflict examples and frozen model predictions", fontsize=14)
    fig.subplots_adjust(
        left=0.03, right=0.99, top=0.92, bottom=0.03,
        hspace=0.35, wspace=0.12,
    )
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
    selected_rows, metadata = collect(artifacts_root)
    png_data = render(artifacts_root, selected_rows)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(selected_rows)
    csv_data = buffer.getvalue().encode("utf-8")
    metadata["selected_predictions_csv_sha256"] = hashlib.sha256(csv_data).hexdigest()
    metadata["example_figure_png_sha256"] = hashlib.sha256(png_data).hexdigest()
    metadata_data = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()

    for output_root in (
        artifacts_root / "figures/cue_conflict",
        repo_root / "task1/results/cue_conflict/figures",
    ):
        write_once(output_root / "selected_examples.png", png_data)
        write_once(output_root / "selected_predictions.csv", csv_data)
        write_once(output_root / "selected_examples_record.json", metadata_data)
    print("PASS: three user-approved cue-conflict examples and 12 predictions verified.")
    print("PASS: figure, selected-predictions CSV, and provenance record saved.")


if __name__ == "__main__":
    main()
