"""One-time final Sketch evaluation for the frozen Task 3 checkpoints.

This module may read target labels only after its caller has authenticated the final
experiment lock.  Training, model selection, and source diagnostics never import it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
from pathlib import Path, PurePosixPath

import numpy as np
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from task3.data import CLASSES, SEED, image_transform
from task3.model import PACSClassifier, state_dict_sha256


MODEL_ORDER = (
    "erm",
    "dan_dg_1",
    "sam",
    "dan_dg_0p1",
    "dan_dg_10",
    "dan_dg_floor_0p1",
    "dan_dg_floor_1",
    "dan_dg_floor_10",
)
MAIN_COMPARISON = ("erm", "dan_dg_1", "sam")
ORIGINAL_STRENGTH_STUDY = ("erm", "dan_dg_0p1", "dan_dg_1", "dan_dg_10")
SUPPLEMENTARY_FLOOR_STUDY = (
    "erm",
    "dan_dg_floor_0p1",
    "dan_dg_floor_1",
    "dan_dg_floor_10",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def runtime_identity(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }


def target_class_id(relative_path: str) -> int:
    path = PurePosixPath(relative_path)
    if len(path.parts) < 3 or path.parts[0] != "sketch":
        raise ValueError(f"Unexpected final target path: {relative_path}")
    class_name = path.parent.name
    if class_name not in CLASSES:
        raise ValueError(f"Unknown final target class: {class_name}")
    return CLASSES.index(class_name)


def validate_target_records(records: list[dict]) -> None:
    if len(records) != 3929:
        raise ValueError(f"Expected 3929 Sketch records, found {len(records)}")
    paths: set[str] = set()
    identifiers: set[str] = set()
    supports = [0] * len(CLASSES)
    for record in records:
        if set(record) != {"path", "id"}:
            raise ValueError(f"Invalid target record fields: {record}")
        path = str(record["path"])
        identifier = str(record["id"])
        if path in paths or identifier in identifiers:
            raise ValueError("Duplicate target path or identifier")
        paths.add(path)
        identifiers.add(identifier)
        supports[target_class_id(path)] += 1
    if any(value == 0 for value in supports):
        raise ValueError("At least one PACS class is absent from Sketch")


class FinalSketchDataset(Dataset):
    def __init__(self, root: Path, records: list[dict]) -> None:
        validate_target_records(records)
        self.root = root.expanduser().resolve()
        if not (self.root / "sketch").is_dir():
            raise FileNotFoundError(f"Final Sketch workspace is missing: {self.root}")
        self.records = records
        self.transform = image_transform(training=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        relative = PurePosixPath(record["path"])
        label = target_class_id(record["path"])
        with Image.open(self.root / Path(*relative.parts)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, record["id"], record["path"]


def metrics_from_predictions(truth: np.ndarray, predicted: np.ndarray) -> dict:
    labels = list(range(len(CLASSES)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id, class_name in enumerate(CLASSES):
        support = int(matrix[class_id].sum())
        correct = int(matrix[class_id, class_id])
        per_class[class_name] = {
            "support": support,
            "correct": correct,
            "accuracy": correct / support if support else 0.0,
        }
    return {
        "count": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def select_change_examples(
    model_name: str,
    records: list[dict],
    truth: np.ndarray,
    erm_predictions: np.ndarray,
    model_predictions: np.ndarray,
    erm_confidence: np.ndarray,
    model_confidence: np.ndarray,
    erm_metrics: dict,
    model_metrics: dict,
    limit: int = 3,
) -> tuple[dict, list[dict]]:
    """Select deterministic examples from the largest class gain and loss."""
    deltas = [
        model_metrics["per_class"][name]["accuracy"]
        - erm_metrics["per_class"][name]["accuracy"]
        for name in CLASSES
    ]
    best_id = max(range(len(CLASSES)), key=lambda index: deltas[index])
    worst_id = min(range(len(CLASSES)), key=lambda index: deltas[index])
    summary = {
        "largest_improvement_class": CLASSES[best_id],
        "largest_improvement_delta_accuracy": float(deltas[best_id]),
        "largest_degradation_class": CLASSES[worst_id],
        "largest_degradation_delta_accuracy": float(deltas[worst_id]),
        "tie_policy": "first_in_fixed_class_order",
    }
    examples: list[dict] = []
    cases = (
        (
            "improvement",
            best_id,
            (model_predictions == truth) & (erm_predictions != truth),
        ),
        (
            "degradation",
            worst_id,
            (erm_predictions == truth) & (model_predictions != truth),
        ),
    )
    for change_type, class_id, condition in cases:
        indices = [
            index
            for index in range(len(records))
            if int(truth[index]) == class_id and bool(condition[index])
        ]
        indices.sort(key=lambda index: records[index]["path"])
        for rank, index in enumerate(indices[:limit], start=1):
            examples.append(
                {
                    "model": model_name,
                    "change_type": change_type,
                    "selected_class": CLASSES[class_id],
                    "rank_by_path": rank,
                    "id": records[index]["id"],
                    "path": records[index]["path"],
                    "true_class": CLASSES[int(truth[index])],
                    "erm_prediction": CLASSES[int(erm_predictions[index])],
                    "model_prediction": CLASSES[int(model_predictions[index])],
                    "erm_confidence": float(erm_confidence[index]),
                    "model_confidence": float(model_confidence[index]),
                }
            )
    return summary, examples


def load_frozen_model(
    model_name: str,
    record: dict,
    device: torch.device,
) -> tuple[PACSClassifier, dict]:
    checkpoint = Path(record["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen checkpoint is missing: {checkpoint}")
    actual_hash = sha256_file(checkpoint)
    if actual_hash != record["checkpoint_sha256"]:
        raise RuntimeError(f"Frozen checkpoint changed: {model_name}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != int(record["selected_epoch"]):
        raise RuntimeError(f"Frozen selected epoch changed: {model_name}")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"Frozen checkpoint lacks model state: {model_name}")
    model = PACSClassifier(pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_hash,
        "state_dict_sha256": state_dict_sha256(model.state_dict()),
        "selected_epoch": int(record["selected_epoch"]),
    }


def evaluate_model(
    model: PACSClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    truth: list[int] = []
    predictions: list[int] = []
    confidence: list[float] = []
    identifiers: list[str] = []
    paths: list[str] = []
    model.eval()
    with torch.inference_mode():
        for images, labels, batch_ids, batch_paths in loader:
            logits, _features = model(images.to(device, non_blocking=True))
            probabilities = logits.softmax(dim=1)
            values, predicted = probabilities.max(dim=1)
            truth.extend(int(value) for value in labels.tolist())
            predictions.extend(int(value) for value in predicted.cpu().tolist())
            confidence.extend(float(value) for value in values.cpu().tolist())
            identifiers.extend(str(value) for value in batch_ids)
            paths.extend(str(value) for value in batch_paths)
    return (
        np.asarray(truth, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        np.asarray(confidence, dtype=np.float64),
        identifiers,
        paths,
    )


def run_final_evaluation(
    lock: dict,
    sketch_root: Path,
    target_records: list[dict],
    output_dir: Path,
    device: torch.device,
    task2_results: dict,
    task2_results_identity: dict,
    task2_predictions: list[dict],
    task2_predictions_identity: dict,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Final output already exists: {output_dir}")
    if tuple(lock.get("model_order", ())) != MODEL_ORDER:
        raise RuntimeError("Final lock model order differs")
    if tuple(lock.get("main_comparison", ())) != MAIN_COMPARISON:
        raise RuntimeError("Final lock main comparison differs")
    validate_target_records(target_records)
    seed_everything(SEED)
    dataset = FinalSketchDataset(sketch_root, target_records)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    output_dir.mkdir(parents=True)
    results: dict[str, dict] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    confidence_by_model: dict[str, np.ndarray] = {}
    reference_truth: np.ndarray | None = None
    reference_ids: list[str] | None = None
    reference_paths: list[str] | None = None

    for model_name in MODEL_ORDER:
        model, identity = load_frozen_model(
            model_name, lock["checkpoints"][model_name], device
        )
        truth, predicted, confidence, identifiers, paths = evaluate_model(
            model, loader, device
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if reference_truth is None:
            reference_truth, reference_ids, reference_paths = truth, identifiers, paths
        elif not (
            np.array_equal(reference_truth, truth)
            and reference_ids == identifiers
            and reference_paths == paths
        ):
            raise RuntimeError("Target evaluation order changed between models")
        results[model_name] = {
            "identity": identity,
            "source_validation": lock["checkpoints"][model_name]["source_validation"],
            "source_diagnostics": lock["checkpoints"][model_name]["source_diagnostics"],
            "target": metrics_from_predictions(truth, predicted),
        }
        predictions_by_model[model_name] = predicted
        confidence_by_model[model_name] = confidence

    assert reference_truth is not None and reference_ids is not None
    assert reference_paths is not None
    erm_accuracy = results["erm"]["target"]["accuracy"]
    for model_name in MODEL_ORDER:
        results[model_name]["target"]["delta_accuracy_vs_erm"] = float(
            results[model_name]["target"]["accuracy"] - erm_accuracy
        )

    prediction_rows: list[dict] = []
    for index, (identifier, path) in enumerate(zip(reference_ids, reference_paths)):
        row: dict[str, object] = {
            "id": identifier,
            "path": path,
            "true_class_id": int(reference_truth[index]),
            "true_class": CLASSES[int(reference_truth[index])],
        }
        for model_name in MODEL_ORDER:
            predicted = int(predictions_by_model[model_name][index])
            row[f"{model_name}_prediction"] = CLASSES[predicted]
            row[f"{model_name}_correct"] = predicted == int(reference_truth[index])
            row[f"{model_name}_confidence"] = float(
                confidence_by_model[model_name][index]
            )
        prediction_rows.append(row)
    write_csv(output_dir / "target_predictions.csv", prediction_rows)

    class_changes: dict[str, dict] = {}
    selected_examples: list[dict] = []
    for model_name in MODEL_ORDER[1:]:
        summary, examples = select_change_examples(
            model_name,
            target_records,
            reference_truth,
            predictions_by_model["erm"],
            predictions_by_model[model_name],
            confidence_by_model["erm"],
            confidence_by_model[model_name],
            results["erm"]["target"],
            results[model_name]["target"],
        )
        class_changes[model_name] = summary
        selected_examples.extend(examples)
    task2_by_id = {row["id"]: row for row in task2_predictions}
    if len(task2_by_id) != len(target_records):
        raise RuntimeError("Task 2 prediction evidence does not cover all target IDs")
    for example in selected_examples:
        prior = task2_by_id.get(example["id"])
        if prior is None:
            raise RuntimeError("A selected example is absent from Task 2 evidence")
        if prior["path"] != example["path"] or prior["true_class"] != example["true_class"]:
            raise RuntimeError("Task 2 and Task 3 target identities disagree")
        example.update(
            {
                "task2_source_only_prediction": prior["source_only_prediction"],
                "task2_source_only_correct": prior["source_only_correct"],
                "task2_dan_1_prediction": prior["dan_1_prediction"],
                "task2_dan_1_correct": prior["dan_1_correct"],
            }
        )
    if selected_examples:
        write_csv(output_dir / "selected_change_examples.csv", selected_examples)

    task2_comparison = {
        "results_evidence": task2_results_identity,
        "predictions_evidence": task2_predictions_identity,
        "target_aware_dan_task2": task2_results["runs"]["dan_1"]["target"],
        "target_free_dan_dg_task3": results["dan_dg_1"]["target"],
        "accuracy_difference_task3_minus_task2": float(
            results["dan_dg_1"]["target"]["accuracy"]
            - task2_results["runs"]["dan_1"]["target"]["accuracy"]
        ),
        "macro_f1_difference_task3_minus_task2": float(
            results["dan_dg_1"]["target"]["macro_f1"]
            - task2_results["runs"]["dan_1"]["target"]["macro_f1"]
        ),
    }
    final = {
        "status": "TASK3_FINAL_SKETCH_EVALUATION_PASS",
        "protocol_version": lock["protocol_version"],
        "experiment_lock_sha256": lock["lock_sha256"],
        "target_evaluation_once": True,
        "target_labels_used_for_training_or_selection": False,
        "target_count": len(target_records),
        "transform": "resize_256_center_crop_224_imagenet_normalization",
        "model_order": list(MODEL_ORDER),
        "main_comparison": list(MAIN_COMPARISON),
        "original_strength_study": list(ORIGINAL_STRENGTH_STUDY),
        "supplementary_floor_study": list(SUPPLEMENTARY_FLOOR_STUDY),
        "runtime": runtime_identity(device),
        "results": results,
        "class_change_summary": class_changes,
        "task2_comparison": task2_comparison,
    }
    atomic_write_json(final, output_dir / "final_results.json")
    return final
