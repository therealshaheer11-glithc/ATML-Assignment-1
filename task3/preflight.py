"""Read-only Task 3 code, source data, initialization, and ERM checks."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision

from task3.config import (
    RUNS,
    code_tree_sha256,
    load_config,
    sha256_file,
    verify_preregistration,
)
from task3.data import (
    EXPECTED_SOURCE_PROTOCOL_SHA256,
    load_source_protocol,
    steps_per_source_epoch,
    verify_source_snapshot,
)
from task3.model import state_dict_sha256


EXPECTED_SHARED_MMD_SHA256 = (
    "cfe0b1d9c22d7f492ea5e8f76732fbabf21c86cb53f24759af65fda09f9bfbcc"
)
EXPECTED_INITIALIZATION_STATE_SHA256 = (
    "4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3"
)
EXPECTED_ERM_CHECKPOINT_SHA256 = (
    "3d28a223e4b97b323cb3a20dcb5b7577af96631f2e6ef1f2bc99d53d85761327"
)
EXPECTED_ERM_SOURCE_VALIDATION = {
    "photo_accuracy": 0.9730538922155688,
    "photo_macro_f1": 0.9681339341150288,
    "art_painting_accuracy": 0.9097560975609756,
    "art_painting_macro_f1": 0.9119171075935996,
    "cartoon_accuracy": 0.9402985074626866,
    "cartoon_macro_f1": 0.947827661029101,
    "mean_source_accuracy": 0.9410361657464104,
    "mean_source_macro_f1": 0.9426262342459099,
}
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


def runtime_identity() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    actual_code_root = Path(__file__).resolve().parents[1]
    if args.code_root.resolve() != actual_code_root:
        raise RuntimeError(
            "--code-root does not identify the repository that supplied task3: "
            f"{args.code_root.resolve()} != {actual_code_root}"
        )

    actual_runtime = runtime_identity()
    if actual_runtime != EXPECTED_RUNTIME:
        raise RuntimeError(
            "Runtime differs from the locked Task 2/3 environment.\n"
            f"Expected: {EXPECTED_RUNTIME}\nActual:   {actual_runtime}"
        )

    configs = {run_id: load_config(run_id) for run_id in RUNS}
    preregistration_sha256 = verify_preregistration(args.preregistration)
    code_sha256 = code_tree_sha256(actual_code_root)

    shared_mmd_path = args.code_root / "shared" / "mmd.py"
    if not shared_mmd_path.is_file():
        raise FileNotFoundError(f"Locked shared MMD is missing: {shared_mmd_path}")
    shared_mmd_sha256 = sha256_file(shared_mmd_path)
    if shared_mmd_sha256 != EXPECTED_SHARED_MMD_SHA256:
        raise RuntimeError(
            f"Shared MMD changed: {shared_mmd_sha256} != {EXPECTED_SHARED_MMD_SHA256}"
        )

    protocol = load_source_protocol(args.protocol)
    dataset = verify_source_snapshot(args.pacs_source_root, protocol)
    steps = steps_per_source_epoch(protocol)
    if steps != 235:
        raise RuntimeError(f"Unexpected source steps per epoch: {steps}")

    initialization = torch.load(
        args.initialization,
        map_location="cpu",
        weights_only=False,
    )
    if "state_dict" not in initialization:
        raise RuntimeError("Common initialization has no state_dict")
    if initialization.get("seed") != 6304:
        raise RuntimeError("Common initialization has the wrong seed")
    if initialization.get("weights") != "ResNet18_Weights.IMAGENET1K_V1":
        raise RuntimeError("Common initialization has the wrong pretrained weights")
    initialization_state_sha256 = state_dict_sha256(initialization["state_dict"])
    if initialization_state_sha256 != EXPECTED_INITIALIZATION_STATE_SHA256:
        raise RuntimeError("Common initialization state differs from Task 2")
    if initialization.get("state_dict_sha256") != EXPECTED_INITIALIZATION_STATE_SHA256:
        raise RuntimeError("Common initialization internal hash is invalid")

    erm_file_sha256 = sha256_file(args.erm_checkpoint)
    if erm_file_sha256 != EXPECTED_ERM_CHECKPOINT_SHA256:
        raise RuntimeError("ERM checkpoint file differs from the selected Task 2 model")
    erm = torch.load(args.erm_checkpoint, map_location="cpu", weights_only=False)
    if erm.get("target_labels_used") is not False:
        raise RuntimeError("ERM checkpoint does not certify target-label exclusion")
    if erm.get("epoch") != 4:
        raise RuntimeError("ERM selected epoch differs from the approved epoch")
    erm_config = erm.get("identity", {}).get("config", {})
    if erm_config.get("run_id") != "source_only":
        raise RuntimeError("The supplied ERM checkpoint is not the Source-only run")
    if erm_config.get("method") != "source_only":
        raise RuntimeError("The supplied ERM checkpoint has the wrong method")
    if (
        erm.get("identity", {}).get("initialization_sha256")
        != EXPECTED_INITIALIZATION_STATE_SHA256
    ):
        raise RuntimeError("ERM checkpoint used a different initialization")
    if (
        erm.get("identity", {}).get("protocol_sha256")
        != "e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74"
    ):
        raise RuntimeError("ERM checkpoint used a different Task 2 source protocol")
    actual_source_validation = erm.get("source_validation", {})
    for name, expected in EXPECTED_ERM_SOURCE_VALIDATION.items():
        actual = actual_source_validation.get(name)
        if actual is None or abs(float(actual) - expected) > 1e-12:
            raise RuntimeError(
                f"ERM source-validation mismatch for {name}: "
                f"{actual} != {expected}"
            )

    result = {
        "status": "TASK3_CODE_PREFLIGHT_PASS",
        "training_started": False,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "runtime": actual_runtime,
        "configs": configs,
        "code_tree_sha256": code_sha256,
        "shared_mmd": {
            "path": str(shared_mmd_path.resolve()),
            "sha256": shared_mmd_sha256,
        },
        "preregistration_sha256": preregistration_sha256,
        "source_protocol_sha256": EXPECTED_SOURCE_PROTOCOL_SHA256,
        "source_dataset": dataset,
        "steps_per_source_epoch": steps,
        "initialization_state_sha256": initialization_state_sha256,
        "erm_checkpoint_file_sha256": erm_file_sha256,
        "erm_selected_epoch": erm["epoch"],
        "erm_source_validation": erm["source_validation"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing != result:
            raise RuntimeError(f"Different preflight output already exists: {args.output}")
    else:
        atomic_write_json(result, args.output)

    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--pacs-source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--erm-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
