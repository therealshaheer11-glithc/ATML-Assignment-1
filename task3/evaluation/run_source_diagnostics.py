"""Run the locked source-only Task 3 diagnostics without accessing Sketch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score

from task3.config import code_tree_sha256
from task3.data import (
    CLASSES,
    SEED,
    SOURCES,
    load_source_protocol,
    make_validation_loader,
    verify_source_snapshot,
)
from task3.evaluation.sharpness import (
    SHARPNESS_RADIUS,
    common_sharpness_proxy,
    select_sharpness_records,
)
from task3.evaluation.source_domain_separability import (
    fit_domain_probe,
    make_probe_partition,
    select_probe_records,
)
from task3.model import PACSClassifier, state_dict_sha256


PROTOCOL_VERSION = "task3-approved-2026-09-24-v1"
TRAINING_CODE_SHA256 = (
    "4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44"
)
EXPECTED_RUNTIME = {
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "numpy": "2.1.3",
    "sklearn": "1.6.1",
    "cuda": "12.8",
    "cudnn": 91900,
    "device_type": "cuda",
    "gpu": "Tesla T4",
}

MODEL_SPECS = {
    "erm": {
        "file_sha256": (
            "3d28a223e4b97b323cb3a20dcb5b7577af96631f2e6ef1f2bc99d53d85761327"
        ),
        "epoch": 4,
        "source_validation": {
            "photo_accuracy": 0.9730538922155688,
            "photo_macro_f1": 0.9681339341150288,
            "art_painting_accuracy": 0.9097560975609756,
            "art_painting_macro_f1": 0.9119171075935996,
            "cartoon_accuracy": 0.9402985074626866,
            "cartoon_macro_f1": 0.947827661029101,
            "mean_source_accuracy": 0.9410361657464104,
            "mean_source_macro_f1": 0.9426262342459099,
        },
    },
    "dan_dg_1": {
        "file_sha256": (
            "a44bff8e519134459c2d6bf056785801f9944945d160e516b4da7f5e2752992b"
        ),
        "epoch": 3,
        "source_validation": {
            "photo_accuracy": 0.9461077844311377,
            "photo_macro_f1": 0.9342793504172562,
            "art_painting_accuracy": 0.8048780487804879,
            "art_painting_macro_f1": 0.8051965091232748,
            "cartoon_accuracy": 0.8699360341151386,
            "cartoon_macro_f1": 0.8685100408435343,
            "mean_source_accuracy": 0.8736406224422547,
            "mean_source_macro_f1": 0.8693286334613551,
            "worst_source_accuracy": 0.8048780487804879,
            "worst_source_macro_f1": 0.8051965091232748,
        },
    },
    "sam": {
        "file_sha256": (
            "040a80ce15974d71f3a90b5aab5050965d27212822079e5a075c742052c0827d"
        ),
        "epoch": 5,
        "source_validation": {
            "photo_accuracy": 0.9790419161676647,
            "photo_macro_f1": 0.9757921211248675,
            "art_painting_accuracy": 0.9243902439024391,
            "art_painting_macro_f1": 0.9236855934108259,
            "cartoon_accuracy": 0.9573560767590619,
            "cartoon_macro_f1": 0.9626644675708457,
            "mean_source_accuracy": 0.9535960789430552,
            "mean_source_macro_f1": 0.9540473940355131,
            "worst_source_accuracy": 0.9243902439024391,
            "worst_source_macro_f1": 0.9236855934108259,
        },
    },
}


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


def write_once_or_verify(payload: dict, path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"Existing diagnostic design differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
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
        "gpu": torch.cuda.get_device_name(0),
    }


def load_model(path: Path, model_name: str, device: torch.device) -> tuple[PACSClassifier, dict]:
    spec = MODEL_SPECS[model_name]
    if not path.is_file() or sha256_file(path) != spec["file_sha256"]:
        raise RuntimeError(f"{model_name} checkpoint identity differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("epoch") != spec["epoch"]:
        raise RuntimeError(f"{model_name} checkpoint selected epoch differs")
    if model_name == "erm":
        if payload.get("target_labels_used") is not False:
            raise RuntimeError("ERM checkpoint does not certify target-label exclusion")
    else:
        if payload.get("source_only_phase") is not True:
            raise RuntimeError(f"{model_name} checkpoint is not source-only")
        if payload.get("sketch_images_accessed") != 0:
            raise RuntimeError(f"{model_name} checkpoint reports Sketch access")
        if payload.get("identity", {}).get("code_sha256") != TRAINING_CODE_SHA256:
            raise RuntimeError(f"{model_name} checkpoint used different training code")

    recorded = payload.get("source_validation", {})
    for key, expected in spec["source_validation"].items():
        if key not in recorded or abs(float(recorded[key]) - float(expected)) > 1e-12:
            raise RuntimeError(f"{model_name} recorded source metric differs: {key}")

    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"{model_name} checkpoint has no model state")
    model = PACSClassifier(pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, {
        "path": str(path),
        "file_sha256": spec["file_sha256"],
        "state_dict_sha256": state_dict_sha256(model.state_dict()),
        "selected_epoch": spec["epoch"],
    }


def validate_sources(model: PACSClassifier, protocol: dict, root: Path, device: torch.device) -> dict:
    metrics: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for domain in SOURCES:
            truth: list[int] = []
            predictions: list[int] = []
            loader = make_validation_loader(
                root,
                protocol["source_splits"][domain]["validation"],
                batch_size=64,
                num_workers=2,
                pin_memory=True,
            )
            for images, labels, identifiers in loader:
                if any(not value.startswith(domain + "/") for value in identifiers):
                    raise RuntimeError("Validation loader crossed source domains")
                logits, _features = model(images.to(device, non_blocking=True))
                truth.extend(int(value) for value in labels.tolist())
                predictions.extend(int(value) for value in logits.argmax(1).cpu().tolist())
            metrics[f"{domain}_accuracy"] = float(accuracy_score(truth, predictions))
            metrics[f"{domain}_macro_f1"] = float(
                f1_score(
                    truth,
                    predictions,
                    labels=list(range(len(CLASSES))),
                    average="macro",
                    zero_division=0,
                )
            )
    metrics["mean_source_accuracy"] = float(
        np.mean([metrics[f"{domain}_accuracy"] for domain in SOURCES])
    )
    metrics["mean_source_macro_f1"] = float(
        np.mean([metrics[f"{domain}_macro_f1"] for domain in SOURCES])
    )
    metrics["worst_source_accuracy"] = float(
        min(metrics[f"{domain}_accuracy"] for domain in SOURCES)
    )
    metrics["worst_source_macro_f1"] = float(
        min(metrics[f"{domain}_macro_f1"] for domain in SOURCES)
    )
    return metrics


def assert_expected_metrics(model_name: str, metrics: dict) -> None:
    for key, expected in MODEL_SPECS[model_name]["source_validation"].items():
        if abs(float(metrics[key]) - float(expected)) > 1e-12:
            raise RuntimeError(f"{model_name} re-evaluation differs for {key}")


def extract_probe_features(
    model: PACSClassifier,
    selected: dict[str, list[dict]],
    expected_identifiers: list[str],
    root: Path,
    device: torch.device,
) -> np.ndarray:
    matrices: list[np.ndarray] = []
    observed_identifiers: list[str] = []
    model.eval()
    with torch.inference_mode():
        for domain in SOURCES:
            loader = make_validation_loader(
                root,
                selected[domain],
                batch_size=64,
                num_workers=2,
                pin_memory=True,
            )
            for images, _labels, identifiers in loader:
                _logits, features = model(images.to(device, non_blocking=True))
                matrices.append(features.cpu().numpy())
                observed_identifiers.extend(str(value) for value in identifiers)
    if observed_identifiers != expected_identifiers:
        raise RuntimeError("Probe feature order differs from the saved identifier order")
    features = np.concatenate(matrices, axis=0)
    if features.shape != (1002, 512) or not np.isfinite(features).all():
        raise RuntimeError(f"Invalid source-domain probe features: {features.shape}")
    return features


def load_sharpness_batch(
    selected: dict[str, list[dict]],
    root: Path,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    image_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    identifiers: list[str] = []
    for domain in SOURCES:
        loader = make_validation_loader(
            root,
            selected[domain],
            batch_size=32,
            num_workers=2,
            pin_memory=False,
        )
        batches = list(loader)
        if len(batches) != 1:
            raise RuntimeError("Sharpness selection did not produce one batch per domain")
        images, labels, paths = batches[0]
        image_parts.append(images)
        label_parts.append(labels)
        identifiers.extend(str(value) for value in paths)
    return torch.cat(image_parts), torch.cat(label_parts), identifiers


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Source diagnostics already exist: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("The locked diagnostic environment requires CUDA")
    device = torch.device("cuda")
    environment = runtime_identity(device)
    if environment != EXPECTED_RUNTIME:
        raise RuntimeError(f"Diagnostic runtime differs: {environment}")
    seed_everything(SEED)

    protocol = load_source_protocol(args.protocol)
    dataset = verify_source_snapshot(args.pacs_source_root, protocol)
    if dataset.get("sketch_images_accessed") != 0:
        raise RuntimeError("Source snapshot verification reports Sketch access")

    probe_selected = select_probe_records(protocol)
    probe_partition = make_probe_partition(probe_selected)
    probe_design = {
        "status": "TASK3_SOURCE_DOMAIN_PROBE_DESIGN_LOCKED",
        "protocol_version": PROTOCOL_VERSION,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "selection_rng": "numpy.default_rng_PCG64_single_stream_domain_order_P_A_C",
        "feature_representation": "raw_unnormalized_512d_pre_classifier",
        "partition": probe_partition,
    }
    probe_design_path = args.output.parent / "source_domain_probe_design.json"
    write_once_or_verify(probe_design, probe_design_path)

    sharpness_selected = select_sharpness_records(protocol)
    sharpness_design = {
        "status": "TASK3_SHARPNESS_BATCH_DESIGN_LOCKED",
        "protocol_version": PROTOCOL_VERSION,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "selection_rng": "numpy.default_rng_PCG64_independent_stream_domain_order_P_A_C",
        "seed": SEED,
        "per_domain": 32,
        "radius": SHARPNESS_RADIUS,
        "records": {
            domain: [dict(record) for record in sharpness_selected[domain]]
            for domain in SOURCES
        },
    }
    sharpness_design_path = args.output.parent / "sharpness_batch_design.json"
    write_once_or_verify(sharpness_design, sharpness_design_path)
    sharpness_images, sharpness_labels, sharpness_identifiers = load_sharpness_batch(
        sharpness_selected, args.pacs_source_root
    )
    expected_sharpness_identifiers = [
        record["path"]
        for domain in SOURCES
        for record in sharpness_selected[domain]
    ]
    if sharpness_identifiers != expected_sharpness_identifiers:
        raise RuntimeError("Sharpness batch order differs from its saved design")

    checkpoint_paths = {
        "erm": args.erm_checkpoint,
        "dan_dg_1": args.training_root / "dan_dg_1" / "best.pt",
        "sam": args.training_root / "sam" / "best.pt",
    }
    model_results = {}
    for model_name in ("erm", "dan_dg_1", "sam"):
        seed_everything(SEED)
        model, checkpoint_identity = load_model(
            checkpoint_paths[model_name], model_name, device
        )
        source_validation = validate_sources(
            model, protocol, args.pacs_source_root, device
        )
        assert_expected_metrics(model_name, source_validation)
        features = extract_probe_features(
            model,
            probe_selected,
            probe_partition["identifiers"],
            args.pacs_source_root,
            device,
        )
        probe = fit_domain_probe(features, probe_partition)
        before_sharpness_state = state_dict_sha256(model.state_dict())
        sharpness = common_sharpness_proxy(
            model,
            sharpness_images.to(device, non_blocking=True),
            sharpness_labels.to(device, non_blocking=True),
        )
        after_sharpness_state = state_dict_sha256(model.state_dict())
        if before_sharpness_state != after_sharpness_state:
            raise RuntimeError(f"{model_name} changed during sharpness measurement")
        model_results[model_name] = {
            "checkpoint": checkpoint_identity,
            "source_validation_reproduced": source_validation,
            "source_domain_separability": probe,
            "common_sharpness_proxy": sharpness,
        }
        del model, features
        torch.cuda.empty_cache()

    result = {
        "status": "TASK3_SOURCE_DIAGNOSTICS_PASS",
        "protocol_version": PROTOCOL_VERSION,
        "source_only_phase": True,
        "diagnostics_completed": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
        "training_code_tree_sha256": TRAINING_CODE_SHA256,
        "diagnostic_code_tree_sha256": code_tree_sha256(args.code_root),
        "source_protocol_sha256": sha256_file(args.protocol),
        "source_snapshot_sha256": dataset["source_snapshot_sha256"],
        "runtime": environment,
        "probe_design": {
            "path": str(probe_design_path),
            "sha256": sha256_file(probe_design_path),
            "image_count": len(probe_partition["identifiers"]),
        },
        "sharpness_batch_design": {
            "path": str(sharpness_design_path),
            "sha256": sha256_file(sharpness_design_path),
            "image_count": len(sharpness_identifiers),
        },
        "models": model_results,
    }
    atomic_write_json(result, args.output)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--pacs-source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--erm-checkpoint", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
