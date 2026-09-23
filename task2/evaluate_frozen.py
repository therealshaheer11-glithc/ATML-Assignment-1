"""Final Task 2 evaluation. This is the only v5 code that reads Sketch labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/task2-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from shared.pacs import (
    CLASSES,
    SEED,
    SOURCES,
    image_transform,
    load_protocol,
    make_validation_loader,
    resolve_root,
    verify_dataset_snapshot,
)
from task2.model import PACSClassifier


ALL_RUNS = ("source_only", "dan_0p1", "dan_1", "dan_10", "dann", "cdan")
MAIN_RUNS = ("source_only", "dan_1", "dann", "cdan")
STRENGTH_RUNS = ("source_only", "dan_0p1", "dan_1", "dan_10")
METHOD_NAMES = {
    "source_only": "Source-only",
    "dan_0p1": "DAN lambda=0.1",
    "dan_1": "DAN lambda=1",
    "dan_10": "DAN lambda=10",
    "dann": "DANN",
    "cdan": "CDAN",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to create empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_class_id(relative_path: str) -> int:
    """Parse a Sketch class only inside the post-freeze evaluation program."""
    path = Path(relative_path)
    if len(path.parts) < 3 or path.parts[0] != "sketch":
        raise ValueError(f"Unexpected target path: {relative_path}")
    class_name = path.parent.name
    if class_name not in CLASSES:
        raise ValueError(f"Unknown target class folder: {class_name}")
    return CLASSES.index(class_name)


class FinalLabeledSketch(Dataset):
    """Expose Sketch labels only after the frozen-manifest gate has passed."""

    def __init__(self, root: Path, records: list[dict]) -> None:
        self.root = resolve_root(root)
        self.records = records
        self.transform = image_transform(training=False)
        if not records or any(set(row) != {"path", "id"} for row in records):
            raise ValueError("Final target dataset received invalid records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        label = target_class_id(record["path"])
        with Image.open(self.root / record["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, record["id"], record["path"]


def classification_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(range(len(CLASSES))),
                average="macro",
                zero_division=0,
            )
        ),
    }


def dominant_confusion_rows(run_id: str, matrix: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for class_id, class_name in enumerate(CLASSES):
        mistakes = matrix[class_id].copy()
        mistakes[class_id] = 0
        total_errors = int(mistakes.sum())
        if total_errors:
            count = int(mistakes.max())
            confused_ids = np.flatnonzero(mistakes == count)
            confused_name = "|".join(CLASSES[int(index)] for index in confused_ids)
            fraction = count / total_errors
        else:
            confused_name, count, fraction = "", 0, 0.0
        rows.append(
            {
                "run_id": run_id,
                "true_class": class_name,
                "dominant_wrong_prediction": confused_name,
                "dominant_confusion_count": count,
                "total_class_errors": total_errors,
                "fraction_of_class_errors": fraction,
            }
        )
    return rows


def make_probe_partition(source_count: int, target_count: int) -> dict:
    """One shared, balanced, reproducible partition for every frozen model."""
    if source_count <= 1 or target_count < source_count:
        raise ValueError("Insufficient features for the balanced domain probe")
    rng = np.random.default_rng(SEED)
    selected_target = np.sort(
        rng.choice(target_count, size=source_count, replace=False)
    )
    labels = np.concatenate(
        (
            np.zeros(source_count, dtype=np.int64),
            np.ones(source_count, dtype=np.int64),
        )
    )
    all_indices = np.arange(2 * source_count)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    return {
        "selected_target_indices": selected_target.tolist(),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "domain_labels": labels.tolist(),
    }


def core_environment(device: torch.device) -> dict:
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


def validate_freeze(
    freeze_path: Path,
    expected_freeze_sha256: str,
    source_audit_path: Path,
    protocol_path: Path,
    pacs_root: Path,
    device: torch.device,
) -> tuple[dict, dict, dict]:
    actual_freeze_sha256 = sha256_file(freeze_path)
    if actual_freeze_sha256 != expected_freeze_sha256:
        raise RuntimeError(
            f"Freeze hash mismatch: {actual_freeze_sha256} != {expected_freeze_sha256}"
        )
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "FROZEN_BEFORE_TARGET_LABEL_ACCESS":
        raise RuntimeError("The checkpoint freeze is not valid")
    if freeze.get("target_labels_accessed") is not False:
        raise RuntimeError("The freeze does not certify pre-target status")
    if tuple(freeze.get("official_runs", {}).keys()) != ALL_RUNS:
        raise RuntimeError("The freeze does not contain the six ordered official runs")
    if Path(freeze["source_audit_path"]).resolve() != source_audit_path.resolve():
        raise RuntimeError("Source-audit path differs from the freeze")
    if sha256_file(source_audit_path) != freeze["source_audit_sha256"]:
        raise RuntimeError("Source-audit hash differs from the freeze")
    source_audit = json.loads(source_audit_path.read_text())
    if source_audit.get("status") != "SOURCE_CHECKPOINT_AUDIT_PASS":
        raise RuntimeError("Source checkpoint audit did not pass")
    if source_audit.get("target_labels_accessed") is not False:
        raise RuntimeError("Source audit target-label flag failed")

    protocol = load_protocol(protocol_path)
    dataset = verify_dataset_snapshot(pacs_root, protocol)
    common_identity = freeze["common_identity"]
    if sha256_file(protocol_path) != common_identity["protocol_sha256"]:
        raise RuntimeError("Protocol hash differs from the frozen identity")
    if dataset["file_list_sha256"] != common_identity["dataset_file_list_sha256"]:
        raise RuntimeError("Dataset snapshot differs from the frozen identity")
    if core_environment(device) != common_identity["environment"]:
        raise RuntimeError("Evaluation environment differs from the frozen environment")

    for run_id, record in freeze["official_runs"].items():
        checkpoint_path = Path(record["checkpoint_path"])
        if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen checkpoint changed: {run_id}")
    return freeze, source_audit, protocol


def extract_source_features(
    model: PACSClassifier,
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[np.ndarray, list[dict]]:
    arrays: list[np.ndarray] = []
    metadata: list[dict] = []
    model.eval()
    with torch.inference_mode():
        for domain in SOURCES:
            for images, labels, paths in loaders[domain]:
                _, features = model(images.to(device, non_blocking=True))
                arrays.append(features.float().cpu().numpy())
                for label, path in zip(labels.tolist(), paths):
                    metadata.append(
                        {
                            "origin": "source",
                            "domain": domain,
                            "path": path,
                            "class_id": int(label),
                        }
                    )
    return np.concatenate(arrays, axis=0), metadata


def extract_target(
    model: PACSClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    feature_arrays: list[np.ndarray] = []
    logit_arrays: list[np.ndarray] = []
    labels: list[int] = []
    identifiers: list[str] = []
    paths: list[str] = []
    model.eval()
    with torch.inference_mode():
        for images, truth, batch_ids, batch_paths in loader:
            logits, features = model(images.to(device, non_blocking=True))
            feature_arrays.append(features.float().cpu().numpy())
            logit_arrays.append(logits.float().cpu().numpy())
            labels.extend(int(value) for value in truth.tolist())
            identifiers.extend(batch_ids)
            paths.extend(batch_paths)
    return (
        np.concatenate(feature_arrays, axis=0),
        np.concatenate(logit_arrays, axis=0),
        np.asarray(labels, dtype=np.int64),
        identifiers,
        paths,
    )


def run_probe(
    source_features: np.ndarray,
    target_features: np.ndarray,
    partition: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    selected_target = np.asarray(partition["selected_target_indices"], dtype=np.int64)
    features = np.concatenate((source_features, target_features[selected_target]), axis=0)
    labels = np.asarray(partition["domain_labels"], dtype=np.int64)
    train_indices = np.asarray(partition["train_indices"], dtype=np.int64)
    test_indices = np.asarray(partition["test_indices"], dtype=np.int64)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(features[train_indices])
    scaled_test = scaler.transform(features[test_indices])
    probe = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    probe.fit(scaled_train, labels[train_indices])
    predicted = probe.predict(scaled_test)
    probabilities = probe.predict_proba(scaled_test)[:, 1]
    matrix = confusion_matrix(labels[test_indices], predicted, labels=[0, 1])
    result = {
        "accuracy": float(accuracy_score(labels[test_indices], predicted)),
        "chance_accuracy": 0.5,
        "source_count": int(len(source_features)),
        "target_count": int(len(selected_target)),
        "train_count": int(len(train_indices)),
        "test_count": int(len(test_indices)),
        "scaler_fit_on_probe_train_only": True,
        "logistic_regression": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": SEED,
        },
        "confusion_matrix_source0_target1": matrix.tolist(),
    }
    return result, predicted, probabilities


def create_plots(
    output: Path,
    main_rows: list[dict],
    strength_rows: list[dict],
    class_change_rows: list[dict],
    matrices: dict[str, list[list[int]]],
    freeze: dict,
) -> None:
    plots = output / "plots"
    plots.mkdir()

    # Main aggregate comparison.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    labels = [METHOD_NAMES[row["run_id"]] for row in main_rows]
    for axis, field, title in zip(
        axes,
        ("target_accuracy", "target_macro_f1", "domain_separability"),
        ("Sketch accuracy", "Sketch macro-F1", "Domain separability"),
    ):
        values = [float(row[field]) for row in main_rows]
        axis.bar(labels, values)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "main_comparison.png", dpi=200)
    plt.close(fig)

    # Domain separability versus target recognition, all six runs.
    fig, axis = plt.subplots(figsize=(6.2, 4.5))
    all_rows = {row["run_id"]: row for row in main_rows + strength_rows}
    for run_id in ALL_RUNS:
        row = all_rows[run_id]
        axis.scatter(row["domain_separability"], row["target_accuracy"], s=50)
        axis.annotate(METHOD_NAMES[run_id], (row["domain_separability"], row["target_accuracy"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("Source-vs-Sketch separability accuracy")
    axis.set_ylabel("Sketch accuracy")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "separability_vs_target_accuracy.png", dpi=200)
    plt.close(fig)

    # DAN strength study (exclude Source-only from the connected lambda curve).
    dan_rows = [row for row in strength_rows if row["run_id"] != "source_only"]
    lambdas = [0.1, 1.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for axis, field, title in zip(
        axes,
        ("mean_source_macro_f1", "domain_separability", "target_accuracy"),
        ("Mean source macro-F1", "Domain separability", "Sketch accuracy"),
    ):
        axis.plot(lambdas, [float(row[field]) for row in dan_rows], marker="o")
        axis.axhline(float(strength_rows[0][field]), color="gray", linestyle="--", label="Source-only")
        axis.set_xscale("log")
        axis.set_xticks(lambdas, ["0.1", "1", "10"])
        axis.set_xlabel("DAN lambda_MMD")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "dan_strength_study.png", dpi=200)
    plt.close(fig)

    # Main-method per-class changes versus Source-only.
    adapted = ("dan_1", "dann", "cdan")
    width = 0.25
    x = np.arange(len(CLASSES))
    fig, axis = plt.subplots(figsize=(10, 4.2))
    lookup = {(row["run_id"], row["class"]): float(row["accuracy_change_vs_source_only"]) for row in class_change_rows}
    for offset, run_id in enumerate(adapted):
        axis.bar(x + (offset - 1) * width, [lookup[(run_id, name)] for name in CLASSES], width, label=METHOD_NAMES[run_id])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, CLASSES, rotation=25)
    axis.set_ylabel("Sketch class-accuracy change")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "per_class_changes_main_methods.png", dpi=200)
    plt.close(fig)

    # Main-method target confusion matrices.
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for axis, run_id in zip(axes.flat, MAIN_RUNS):
        matrix = np.asarray(matrices[run_id])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(METHOD_NAMES[run_id])
        axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots / "target_confusions_main_methods.png", dpi=200)
    plt.close(fig)

    # Training curves from the exact frozen histories.
    for run_id in ALL_RUNS:
        record = freeze["official_runs"][run_id]
        history_path = Path(record["checkpoint_path"]).parent / "history.csv"
        with history_path.open(newline="") as handle:
            history = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in history]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot(epochs, [float(row["classification_loss"]) for row in history])
        axes[0, 0].set_title("Source classification loss")

        if run_id.startswith("dan_"):
            axes[0, 1].plot(epochs, [float(row["mmd_loss"]) for row in history])
            axes[0, 1].set_title("Unweighted MMD loss")
        elif run_id in {"dann", "cdan"}:
            axes[0, 1].plot(epochs, [float(row["domain_loss"]) for row in history], label="loss")
            axes[0, 1].plot(epochs, [float(row["domain_accuracy"]) for row in history], label="accuracy")
            axes[0, 1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
            axes[0, 1].set_title("Domain discriminator")
            axes[0, 1].legend(fontsize=8)
        else:
            axes[0, 1].axis("off")

        for domain in SOURCES:
            axes[1, 0].plot(epochs, [float(row[f"{domain}_macro_f1"]) for row in history], label=domain)
        axes[1, 0].plot(epochs, [float(row["mean_source_macro_f1"]) for row in history], color="black", linewidth=2, label="mean")
        axes[1, 0].axvline(int(record["selected_epoch"]), color="gray", linestyle="--")
        axes[1, 0].set_title("Source validation macro-F1")
        axes[1, 0].legend(fontsize=7)

        axes[1, 1].plot(epochs, [float(row["gradient_norm"]) for row in history], label="pre-clipping norm")
        axes[1, 1].plot(epochs, [float(row["gradient_clipped_fraction"]) for row in history], label="clipped fraction")
        axes[1, 1].set_yscale("symlog", linthresh=1.0)
        axes[1, 1].set_title("Gradient diagnostics")
        axes[1, 1].legend(fontsize=8)

        for axis in axes.flat:
            if axis.axison:
                axis.set_xlabel("Source epoch")
                axis.grid(alpha=0.2)
        fig.suptitle(METHOD_NAMES[run_id])
        fig.tight_layout()
        fig.savefig(plots / f"{run_id}_training.png", dpi=200)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    freeze, source_audit, protocol = validate_freeze(
        args.freeze,
        args.expected_freeze_sha256,
        args.source_audit,
        args.protocol,
        args.pacs_root,
        device,
    )

    if args.output.exists():
        raise FileExistsError(f"Final output already exists: {args.output}")
    partial = args.output.with_name(args.output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Preserve or remove previous partial output: {partial}")
    partial.mkdir(parents=True)

    # The freeze has passed. Target labels become available only from this point.
    target_dataset = FinalLabeledSketch(args.pacs_root, protocol["target_unlabeled"])
    target_loader = DataLoader(
        target_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    source_loaders = {
        domain: make_validation_loader(
            args.pacs_root,
            protocol["source_splits"][domain]["validation"],
            batch_size=64,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for domain in SOURCES
    }

    source_count = sum(len(protocol["source_splits"][domain]["validation"]) for domain in SOURCES)
    partition = make_probe_partition(source_count, len(target_dataset))
    selected_target = np.asarray(partition["selected_target_indices"], dtype=np.int64)

    target_truth: np.ndarray | None = None
    target_ids: list[str] | None = None
    target_paths: list[str] | None = None
    source_metadata: list[dict] | None = None
    predictions: dict[str, np.ndarray] = {}
    confidences: dict[str, np.ndarray] = {}
    all_results: dict[str, dict] = {}
    matrices: dict[str, list[list[int]]] = {}
    per_class_rows: list[dict] = []
    dominant_rows: list[dict] = []
    failure_rows: list[dict] = []
    probe_prediction_rows: list[dict] = []

    for run_id in ALL_RUNS:
        frozen = freeze["official_runs"][run_id]
        checkpoint_path = Path(frozen["checkpoint_path"])
        if sha256_file(checkpoint_path) != frozen["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed after freeze: {run_id}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("target_labels_used") is not False:
            raise RuntimeError(f"Pre-evaluation target flag failed: {run_id}")

        model = PACSClassifier(pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        source_features, current_source_metadata = extract_source_features(model, source_loaders, device)
        target_features, logits, truth, identifiers, paths = extract_target(model, target_loader, device)

        if target_truth is None:
            target_truth, target_ids, target_paths = truth.copy(), identifiers, paths
            source_metadata = current_source_metadata
        else:
            if not np.array_equal(target_truth, truth) or target_ids != identifiers or target_paths != paths:
                raise RuntimeError("Target order changed between frozen methods")
            if source_metadata != current_source_metadata:
                raise RuntimeError("Source feature order changed between methods")

        predicted = logits.argmax(axis=1)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        confidence = probabilities.max(axis=1)
        predictions[run_id] = predicted
        confidences[run_id] = confidence

        metrics = classification_metrics(truth, predicted)
        matrix = confusion_matrix(truth, predicted, labels=list(range(len(CLASSES))))
        matrices[run_id] = matrix.tolist()

        probe, probe_predicted, probe_probability = run_probe(source_features, target_features, partition)
        test_indices = np.asarray(partition["test_indices"], dtype=np.int64)
        domain_labels = np.asarray(partition["domain_labels"], dtype=np.int64)
        assert source_metadata is not None and target_ids is not None and target_paths is not None
        probe_metadata = source_metadata + [
            {
                "origin": "target",
                "domain": "sketch",
                "path": target_paths[index],
                "id": target_ids[index],
            }
            for index in selected_target
        ]
        for position, combined_index in enumerate(test_indices):
            metadata = probe_metadata[int(combined_index)]
            probe_prediction_rows.append(
                {
                    "run_id": run_id,
                    "combined_index": int(combined_index),
                    "origin": metadata["origin"],
                    "domain": metadata["domain"],
                    "path": metadata["path"],
                    "id": metadata.get("id", ""),
                    "true_domain_label": int(domain_labels[combined_index]),
                    "predicted_domain_label": int(probe_predicted[position]),
                    "probability_target": float(probe_probability[position]),
                }
            )

        class_results: dict[str, dict] = {}
        for class_id, class_name in enumerate(CLASSES):
            mask = truth == class_id
            class_accuracy = float((predicted[mask] == class_id).mean())
            class_results[class_name] = {"support": int(mask.sum()), "accuracy": class_accuracy}
            per_class_rows.append(
                {
                    "run_id": run_id,
                    "method": METHOD_NAMES[run_id],
                    "class": class_name,
                    "support": int(mask.sum()),
                    "target_class_accuracy": class_accuracy,
                }
            )

            mistaken = np.flatnonzero(mask & (predicted != class_id))
            ranked = mistaken[np.argsort(-confidence[mistaken])][:3]
            for rank, index in enumerate(ranked, start=1):
                failure_rows.append(
                    {
                        "run_id": run_id,
                        "true_class": class_name,
                        "rank_within_true_class": rank,
                        "id": identifiers[int(index)],
                        "path": paths[int(index)],
                        "predicted_class": CLASSES[int(predicted[index])],
                        "confidence": float(confidence[index]),
                    }
                )

        dominant_rows.extend(dominant_confusion_rows(run_id, matrix))
        all_results[run_id] = {
            "run_id": run_id,
            "method": METHOD_NAMES[run_id],
            "protocol_version": frozen["protocol_version"],
            "selected_epoch": frozen["selected_epoch"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "source_validation": source_audit["runs"][run_id]["source_domains"],
            "mean_source_validation_macro_f1": source_audit["runs"][run_id]["mean_source_validation_macro_f1"],
            "target": {**metrics, "count": int(len(truth)), "per_class": class_results},
            "domain_probe": probe,
        }
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "target_accuracy": metrics["accuracy"],
                    "target_macro_f1": metrics["macro_f1"],
                    "domain_separability": probe["accuracy"],
                }
            ),
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert target_truth is not None and target_ids is not None and target_paths is not None
    baseline_accuracy = all_results["source_only"]["target"]["accuracy"]
    baseline_per_class = {
        name: all_results["source_only"]["target"]["per_class"][name]["accuracy"]
        for name in CLASSES
    }

    comparison_rows: list[dict] = []
    for run_id in ALL_RUNS:
        result = all_results[run_id]
        source = result["source_validation"]
        row = {
            "run_id": run_id,
            "method": METHOD_NAMES[run_id],
            "protocol_version": result["protocol_version"],
            "selected_epoch": result["selected_epoch"],
        }
        for domain in SOURCES:
            row[f"{domain}_source_accuracy"] = source[domain]["accuracy"]
            row[f"{domain}_source_macro_f1"] = source[domain]["macro_f1"]
        row["mean_source_accuracy"] = float(np.mean([source[domain]["accuracy"] for domain in SOURCES]))
        row["mean_source_macro_f1"] = result["mean_source_validation_macro_f1"]
        row["target_accuracy"] = result["target"]["accuracy"]
        row["target_macro_f1"] = result["target"]["macro_f1"]
        row["target_accuracy_change_vs_source_only"] = result["target"]["accuracy"] - baseline_accuracy
        row["domain_separability"] = result["domain_probe"]["accuracy"]
        comparison_rows.append(row)

    comparison_lookup = {row["run_id"]: row for row in comparison_rows}
    main_rows = [comparison_lookup[run_id] for run_id in MAIN_RUNS]
    strength_rows = [comparison_lookup[run_id] for run_id in STRENGTH_RUNS]

    class_change_rows: list[dict] = []
    for row in per_class_rows:
        baseline = baseline_per_class[row["class"]]
        class_change_rows.append(
            {
                **row,
                "source_only_target_class_accuracy": baseline,
                "accuracy_change_vs_source_only": row["target_class_accuracy"] - baseline,
            }
        )

    change_summary: dict[str, dict] = {}
    for run_id in ALL_RUNS:
        rows = [row for row in class_change_rows if row["run_id"] == run_id]
        best_value = max(float(row["accuracy_change_vs_source_only"]) for row in rows)
        worst_value = min(float(row["accuracy_change_vs_source_only"]) for row in rows)
        change_summary[run_id] = {
            "largest_improvement": {
                "classes": [row["class"] for row in rows if np.isclose(float(row["accuracy_change_vs_source_only"]), best_value, rtol=0.0, atol=1e-15)],
                "change": best_value,
            },
            "largest_degradation": {
                "classes": [row["class"] for row in rows if np.isclose(float(row["accuracy_change_vs_source_only"]), worst_value, rtol=0.0, atol=1e-15)],
                "change": worst_value,
            },
        }

    target_prediction_rows: list[dict] = []
    for index, (identifier, path) in enumerate(zip(target_ids, target_paths)):
        row: dict = {
            "id": identifier,
            "path": path,
            "true_class": CLASSES[int(target_truth[index])],
        }
        for run_id in ALL_RUNS:
            row[f"{run_id}_prediction"] = CLASSES[int(predictions[run_id][index])]
            row[f"{run_id}_correct"] = int(predictions[run_id][index] == target_truth[index])
            row[f"{run_id}_confidence"] = float(confidences[run_id][index])
        target_prediction_rows.append(row)

    write_csv(partial / "all_six_comparison.csv", comparison_rows)
    write_csv(partial / "main_four_comparison.csv", main_rows)
    write_csv(partial / "dan_strength_study.csv", strength_rows)
    write_csv(partial / "per_class_target_accuracy.csv", per_class_rows)
    write_csv(partial / "per_class_changes_vs_source_only.csv", class_change_rows)
    write_csv(partial / "dominant_confusions.csv", dominant_rows)
    write_csv(partial / "selected_failure_examples.csv", failure_rows)
    write_csv(partial / "target_predictions.csv", target_prediction_rows)
    write_csv(partial / "domain_probe_test_predictions.csv", probe_prediction_rows)

    partition_record = {
        **partition,
        "seed": SEED,
        "source_order": list(SOURCES),
        "source_count": source_count,
        "target_total_count": len(target_dataset),
        "target_selected_count": len(selected_target),
        "selected_target_ids": [target_ids[index] for index in selected_target],
        "selected_target_paths": [target_paths[index] for index in selected_target],
    }
    (partial / "domain_probe_partition.json").write_text(json.dumps(partition_record, indent=2) + "\n")
    (partial / "confusion_matrices.json").write_text(json.dumps(matrices, indent=2) + "\n")
    (partial / "class_change_summary.json").write_text(json.dumps(change_summary, indent=2) + "\n")
    (partial / "final_results.json").write_text(
        json.dumps(
            {
                "assignment_main_comparison": list(MAIN_RUNS),
                "controlled_study": list(STRENGTH_RUNS),
                "runs": all_results,
                "target_labels_first_accessed_after_freeze_validation": True,
                "target_labels_used_for_training_or_selection": False,
            },
            indent=2,
        )
        + "\n"
    )

    create_plots(partial, main_rows, strength_rows, class_change_rows, matrices, freeze)

    completion = {
        "status": "FINAL_EVALUATION_COMPLETE",
        "freeze_path": str(args.freeze.resolve()),
        "freeze_sha256": args.expected_freeze_sha256,
        "source_audit_path": str(args.source_audit.resolve()),
        "source_audit_sha256": sha256_file(args.source_audit),
        "protocol_sha256": sha256_file(args.protocol),
        "dataset_file_list_sha256": protocol["file_list_sha256"],
        "runs_evaluated": list(ALL_RUNS),
        "main_comparison": list(MAIN_RUNS),
        "target_count": len(target_dataset),
        "domain_probe_chance": 0.5,
        "target_labels_first_accessed_after_freeze_validation": True,
        "target_labels_used_for_training_or_selection": False,
        "environment": core_environment(device),
    }
    (partial / "evaluation_complete.json").write_text(json.dumps(completion, indent=2) + "\n")

    files = []
    for path in sorted(partial.rglob("*")):
        if path.is_file() and path.name != "EVALUATION-MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(partial).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    (partial / "EVALUATION-MANIFEST.json").write_text(
        json.dumps({"status": "COMPLETE", "files": files}, indent=2) + "\n"
    )
    os.replace(partial, args.output)

    print(json.dumps(completion, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--pacs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
