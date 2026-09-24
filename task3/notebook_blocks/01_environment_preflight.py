"""Task 3 notebook block 01: runtime identity and source-only guardrails.

This block does not mount Drive, inspect PACS, load a model, or access Sketch.
Run it before any other Task 3 block and preserve the emitted JSON file.
"""

import hashlib
import json
import os
import platform
import random
from pathlib import Path, PurePosixPath

# PyTorch requires this setting for deterministic CUDA matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import sklearn
import torch
import torchvision


PROTOCOL_VERSION = "task3-approved-2026-09-24-v1"
SEED = 6304
SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
FORBIDDEN_TARGET_DOMAIN = "sketch"

LOCKED_IDENTITIES = {
    "task2_protocol_sha256": (
        "e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74"
    ),
    "dataset_file_list_sha256": (
        "559ac63b8df8e07330b97112e5ec4c414b3957585b28d21b4cfecd2181f538e0"
    ),
    "common_initialization_state_sha256": (
        "4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3"
    ),
    "erm_checkpoint_file_sha256": (
        "3d28a223e4b97b323cb3a20dcb5b7577af96631f2e6ef1f2bc99d53d85761327"
    ),
}

LOCKED_SOURCE_COUNTS = {
    "photo": {"train": 1336, "validation": 334},
    "art_painting": {"train": 1638, "validation": 410},
    "cartoon": {"train": 1875, "validation": 469},
}

LOCKED_TRAINING = {
    "seed": 6304,
    "weights": "ResNet18_Weights.IMAGENET1K_V1",
    "num_classes": 7,
    "feature_width": 512,
    "epochs": 30,
    "patience": 5,
    "source_batch_per_domain": 8,
    "steps_per_source_epoch": 235,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "gradient_clipping_max_norm": 20.0,
    "selection_metric": "unweighted_mean_source_validation_macro_f1",
    "dan_dg_main_lambda": 1.0,
    "dan_dg_study_lambdas": [0.1, 1.0, 10.0],
    "sam_rho": 0.05,
    "mmd_feature_normalization": "l2_per_sample_mmd_input_only",
    "adversarial_feature_normalization": "not_applicable_to_task3",
    "mmd_bandwidth_pairs": "strict_upper_triangle_keep_off_diagonal_zeros",
    "mmd_kernel": "exp(-squared_distance/(2*bandwidth))",
    "mmd_kernel_factors": [0.5, 1.0, 2.0],
    "mmd_estimator": "v_statistic_include_within_domain_diagonals",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_source_record(record: dict) -> None:
    """Reject any record that is not an approved labeled-source record."""
    if set(record) != {"path", "class_id"}:
        raise ValueError(f"Invalid source-record fields: {sorted(record)}")
    parts = PurePosixPath(str(record["path"])).parts
    if not parts or parts[0] not in SOURCE_DOMAINS:
        raise RuntimeError(f"Target or unknown-domain record rejected: {record['path']}")
    if FORBIDDEN_TARGET_DOMAIN in {part.lower() for part in parts}:
        raise RuntimeError(f"Sketch record rejected during source-only phase: {record['path']}")
    if not isinstance(record["class_id"], int) or not 0 <= record["class_id"] < 7:
        raise ValueError(f"Invalid class ID in source record: {record}")


seed_everything(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "Task 3 training requires a CUDA runtime. In Colab, select a GPU runtime and "
        "rerun this first block before mounting Drive or loading any data."
    )

runtime_identity = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": np.__version__,
    "sklearn": sklearn.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "device_type": "cuda",
    "gpu": torch.cuda.get_device_name(0),
}

preflight_record = {
    "status": "TASK3_BLOCK_01_PASS",
    "protocol_version": PROTOCOL_VERSION,
    "source_only_phase": True,
    "drive_mounted_by_this_block": False,
    "dataset_traversed_by_this_block": False,
    "models_loaded_by_this_block": False,
    "sketch_images_accessed": 0,
    "source_domains": list(SOURCE_DOMAINS),
    "forbidden_target_domain": FORBIDDEN_TARGET_DOMAIN,
    "locked_identities": LOCKED_IDENTITIES,
    "locked_source_counts": LOCKED_SOURCE_COUNTS,
    "locked_training": LOCKED_TRAINING,
    "runtime": runtime_identity,
}

output_path = Path("/content/task3_runtime_preflight.json")
output_path.write_text(json.dumps(preflight_record, indent=2) + "\n")

print(json.dumps(preflight_record, indent=2))
print(f"\nSaved: {output_path}")
print("Next action: inspect this record before mounting Drive or supplying any path.")
