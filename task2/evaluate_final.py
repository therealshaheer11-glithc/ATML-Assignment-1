"""Final Task 2 analysis; this is the only script that reads Sketch labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from shared.pacs import CLASSES, SEED, SOURCES, load_protocol, resolve_root, transform
from task2.freeze import sha256
from task2.model import PACSClassifier
from task2.train import MAIN_RUNS, load_config, make_validation_loader, source_code_sha256, validate_sources, verify_dataset_snapshot


class FinalLabeledSketch(Dataset):
    """Sketch labels become available here, after the freeze manifest passes."""

    def __init__(self, root: Path, records: list[dict]) -> None:
        self.root, self.records, self.image_transform = root, records, transform(False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = Path(record["path"])
        label_name = path.parent.name
        if label_name not in CLASSES:
            raise ValueError(f"Unknown Sketch class folder: {label_name}")
        with Image.open(self.root / path) as image:
            pixels = self.image_transform(image.convert("RGB"))
        return pixels, CLASSES.index(label_name), record["id"]


def extract(model: PACSClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    features, logits, labels, identifiers = [], [], [], []
    with torch.inference_mode():
        for images, truth, ids in loader:
            scores, representation = model(images.to(device, non_blocking=True))
            features.append(representation.float().cpu().numpy())
            logits.append(scores.float().cpu().numpy())
            labels.extend(truth.tolist())
            identifiers.extend(ids)
    return np.concatenate(features), np.concatenate(logits), np.asarray(labels), identifiers


def domain_separability(source_features: dict[str, np.ndarray], target_features: np.ndarray) -> dict:
    """Balanced 70/30 source-versus-target logistic probe, seed 6304."""
    rng = np.random.default_rng(SEED)
    per_source = min(*(len(source_features[domain]) for domain in SOURCES), len(target_features) // 3)
    if per_source < 2:
        raise ValueError("Too few features for a domain-separability probe")
    source = np.concatenate([
        source_features[domain][rng.choice(len(source_features[domain]), per_source, replace=False)]
        for domain in SOURCES
    ])
    target = target_features[rng.choice(len(target_features), len(source), replace=False)]
    feature_matrix = np.concatenate((source, target))
    domain_labels = np.concatenate((np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)))
    train_idx, test_idx = train_test_split(
        np.arange(len(feature_matrix)), test_size=0.3, random_state=SEED,
        stratify=domain_labels,
    )
    probe = LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=SEED)
    probe.fit(feature_matrix[train_idx], domain_labels[train_idx])
    return {
        "accuracy": float(accuracy_score(domain_labels[test_idx], probe.predict(feature_matrix[test_idx]))),
        "source_count": len(source), "target_count": len(target),
        "train_count": len(train_idx), "test_count": len(test_idx),
        "per_source_domain": per_source,
    }


def validate_lock(lock: dict, protocol: dict) -> None:
    if set(lock.get("runs", {})) != set(MAIN_RUNS):
        raise RuntimeError("Final evaluation needs all six frozen runs")
    if lock["protocol_file_list_sha256"] != protocol["file_list_sha256"]:
        raise RuntimeError("PACS snapshot differs from the frozen runs")
    if lock["source_code_sha256"] != source_code_sha256():
        raise RuntimeError("Evaluation code changed after runs were frozen")
    for run_id, record in lock["runs"].items():
        if record["config"] != load_config(run_id):
            raise RuntimeError(f"Config changed after lock for {run_id}")
        if record["checkpoint_sha256"] != sha256(Path(record["directory"]) / "best.pt"):
            raise RuntimeError(f"Checkpoint changed after lock for {run_id}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    lock = json.loads(args.freeze.read_text())
    protocol = load_protocol(args.protocol)
    validate_lock(lock, protocol)
    root = resolve_root(args.pacs_root)
    verify_dataset_snapshot(root, protocol)
    if args.output.exists():
        raise FileExistsError(f"Will not overwrite existing final analysis: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_loaders = {
        domain: make_validation_loader(root, protocol["source_splits"][domain]["validation"], args.num_workers, device)
        for domain in SOURCES
    }
    target_loader = DataLoader(
        FinalLabeledSketch(root, protocol["target_unlabeled"]),
        batch_size=64, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows, predictions, class_metrics = [], {}, {}
    canonical_ids, canonical_truth = None, None
    for run_id in MAIN_RUNS:
        record = lock["runs"][run_id]
        checkpoint = torch.load(Path(record["directory"]) / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != record["config"] or checkpoint["file_list_sha256"] != protocol["file_list_sha256"]:
            raise RuntimeError(f"Invalid checkpoint contents for {run_id}")
        model = PACSClassifier(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state"])
        source_metrics = validate_sources(model, source_loaders, device)
        source_features = {
            domain: extract(model, source_loaders[domain], device)[0]
            for domain in SOURCES
        }
        target_features, target_logits, truth, ids = extract(model, target_loader, device)
        if canonical_ids is None:
            canonical_ids, canonical_truth = ids, truth.copy()
        elif ids != canonical_ids or not np.array_equal(truth, canonical_truth):
            raise RuntimeError("Target evaluation order changed between methods")
        predicted = target_logits.argmax(axis=1)
        predictions[run_id] = predicted
        probe = domain_separability(source_features, target_features)
        result = {
            "run_id": run_id, "method": record["config"]["method"],
            "mmd_lambda": record["config"].get("mmd_lambda", ""),
            "best_epoch": record["best_epoch"], **source_metrics,
            "target_accuracy": float(accuracy_score(truth, predicted)),
            "target_macro_f1": float(f1_score(truth, predicted, labels=list(range(7)), average="macro", zero_division=0)),
            "domain_separability": probe["accuracy"],
        }
        rows.append(result)
        per_class = []
        for class_id, name in enumerate(CLASSES):
            mask = truth == class_id
            per_class.append({"class": name, "support": int(mask.sum()), "accuracy": float((predicted[mask] == truth[mask]).mean())})
        class_metrics[run_id] = per_class
        (args.output / f"{run_id}_class_accuracy.json").write_text(json.dumps(per_class, indent=2) + "\n")
        matrix = confusion_matrix(truth, predicted, labels=list(range(7)))
        (args.output / f"{run_id}_confusion.json").write_text(json.dumps(matrix.tolist(), indent=2) + "\n")
        confusions = []
        for class_id, name in enumerate(CLASSES):
            mistakes = matrix[class_id].copy()
            mistakes[class_id] = 0
            confused_with = int(mistakes.argmax())
            confusions.append({"run_id": run_id, "true_class": name, "most_confused_with": CLASSES[confused_with], "count": int(mistakes[confused_with]), "total_errors": int(mistakes.sum())})
        write_csv(args.output / f"{run_id}_dominant_confusions.csv", confusions)
        (args.output / f"{run_id}_domain_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
        print(json.dumps({"run_id": run_id, "target_accuracy": result["target_accuracy"], "domain_separability": probe["accuracy"]}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline_accuracy = next(row["target_accuracy"] for row in rows if row["run_id"] == "source_only")
    for row in rows:
        row["target_accuracy_change_vs_source_only"] = row["target_accuracy"] - baseline_accuracy
    write_csv(args.output / "comparison.csv", rows)
    baseline_class_accuracy = {item["class"]: item["accuracy"] for item in class_metrics["source_only"]}
    class_change_rows = [
        {"run_id": run_id, "class": item["class"], "support": item["support"],
         "accuracy": item["accuracy"], "source_only_accuracy": baseline_class_accuracy[item["class"]],
         "accuracy_change": item["accuracy"] - baseline_class_accuracy[item["class"]]}
        for run_id in MAIN_RUNS for item in class_metrics[run_id]
    ]
    write_csv(args.output / "class_accuracy_changes.csv", class_change_rows)
    prediction_rows = []
    for i, identifier in enumerate(ids):
        row = {"id": identifier, "true_class": CLASSES[int(truth[i])]}
        for run_id in MAIN_RUNS:
            row[f"{run_id}_prediction"] = CLASSES[int(predictions[run_id][i])]
            row[f"{run_id}_correct"] = int(predictions[run_id][i] == truth[i])
        prediction_rows.append(row)
    write_csv(args.output / "target_predictions.csv", prediction_rows)
    (args.output / "analysis_manifest.json").write_text(json.dumps({"freeze_manifest": str(args.freeze.resolve()), "protocol_file_list_sha256": protocol["file_list_sha256"], "runs_evaluated": list(MAIN_RUNS), "target_labels_used_only_in_final_analysis": True}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--pacs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("shared/splits/pacs_sketch_seed6304.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/task2_final"))
    parser.add_argument("--num-workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
