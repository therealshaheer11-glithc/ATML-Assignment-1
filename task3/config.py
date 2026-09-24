"""Load and enforce the approved Task 3 training configurations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"

RUNS = {
    "dan_dg_0p1": ("dan_dg", 0.1, None),
    "dan_dg_1": ("dan_dg", 1.0, None),
    "dan_dg_10": ("dan_dg", 10.0, None),
    "sam": ("sam", None, 0.05),
}

EXPECTED_PREREGISTRATION_SHA256 = (
    "e8377b762cc1872b6e814e565869414320cff94bbacd69fe097d70b29d4146fb"
)

LOCKED_BASE = {
    "protocol_version": "task3-approved-2026-09-24-v1",
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
    "adam_betas": [0.9, 0.999],
    "adam_epsilon": 1e-8,
    "amsgrad": False,
    "adam_foreach": False,
    "adam_fused": False,
    "scheduler": None,
    "precision": "float32",
    "gradient_clipping": 20.0,
    "gradient_clipping_norm_type": 2.0,
    "gradient_clipping_foreach": False,
    "class_loss_reduction": "mean",
    "label_smoothing": 0.0,
    "class_weights": None,
    "selection_metric": "unweighted_mean_source_validation_macro_f1",
    "tie_policy": "strict_improvement_earliest_tie",
    "batchnorm_running_statistics": "frozen_at_imagenet_values",
    "batchnorm_affine_parameters": "trainable",
    "mmd_feature_normalization": "l2_per_sample_mmd_input_only",
    "adversarial_feature_normalization": "not_applicable_to_task3",
    "mmd_estimator": "literal_empirical_mean_embedding_v_statistic",
    "mmd_bandwidth_pairs": "strict_upper_triangle_keep_off_diagonal_zeros",
    "mmd_kernel": "exp(-squared_distance/(2*bandwidth))",
    "mmd_kernel_factors": [0.5, 1.0, 2.0],
    "mmd_bandwidth_gradient": "detached",
    "mmd_invalid_bandwidth": "stop",
    "dan_dg_pair_reduction": "mean_over_three_unordered_source_pairs",
    "sam_variant": "standard_non_adaptive",
    "sam_first_pass_clipping": False,
    "sam_second_pass_clipping": True,
    "validation_batch_size": 64,
    "num_workers": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_tree_sha256(root: Path) -> str:
    """Hash the executable Task 3 source plus the locked shared MMD implementation."""
    selected: list[Path] = [root / "shared" / "mmd.py"]
    task3_root = root / "task3"
    selected.extend(
        path
        for path in task3_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not {"docs", "notebook_blocks", "provenance", "results"}.intersection(
            path.parts
        )
        and path.suffix in {".py", ".json", ".md"}
    )

    digest = hashlib.sha256()
    for path in sorted(set(selected)):
        if not path.is_file():
            raise FileNotFoundError(f"Required code file is missing: {path}")
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_preregistration(path: Path) -> str:
    actual = sha256_file(path)
    if actual != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            "Task 3 preregistration changed after approval: "
            f"{actual} != {EXPECTED_PREREGISTRATION_SHA256}"
        )
    text = path.read_text().strip()
    if not text or "PENDING" in text.upper() or "REPLACE THIS" in text.upper():
        raise RuntimeError("Task 3 preregistration is incomplete")
    return actual


def load_config(run_id: str) -> dict:
    if run_id not in RUNS:
        raise ValueError(f"Unknown run {run_id!r}; choose one of {tuple(RUNS)}")

    base = json.loads((CONFIG_DIR / "base.json").read_text())
    specific = json.loads((CONFIG_DIR / f"{run_id}.json").read_text())
    config = {**base, **specific}

    for key, expected in LOCKED_BASE.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Locked Task 3 configuration field changed: {key}")

    method, mmd_lambda, sam_rho = RUNS[run_id]
    expected_identity = {
        "run_id": run_id,
        "method": method,
        "mmd_lambda": mmd_lambda,
        "sam_rho": sam_rho,
    }
    for key, expected in expected_identity.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"Run-specific Task 3 field changed: {key}={config.get(key)!r}, "
                f"expected {expected!r}"
            )

    return config
