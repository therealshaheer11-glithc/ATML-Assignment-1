"""Load and reject changes to the locked corrected-run configurations."""

from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
RUNS = {
    "source_only": ("source_only", None),
    "dan_0p1": ("dan", 0.1),
    "dan_1": ("dan", 1.0),
    "dan_10": ("dan", 10.0),
    "dann": ("dann", None),
    "cdan": ("cdan", None),
}


LOCKED_BASE = {
    "seed": 6304,
    "weights": "ResNet18_Weights.IMAGENET1K_V1",
    "num_classes": 7,
    "feature_width": 512,
    "epochs": 30,
    "patience": 5,
    "source_batch_per_domain": 8,
    "target_batch": 24,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "adam_betas": [0.9, 0.999],
    "adam_epsilon": 1e-8,
    "amsgrad": False,
    "adam_foreach": False,
    "adam_fused": False,
    "class_loss_reduction": "mean",
    "domain_loss_reduction": "mean",
    "label_smoothing": 0.0,
    "class_weights": None,
    "scheduler": None,
    "precision": "float32",
    "gradient_clipping": 20.0,
    "gradient_clipping_norm_type": 2.0,
    "gradient_clipping_foreach": False,
    "selection_metric": "unweighted_mean_source_validation_macro_f1",
    "tie_policy": "strict_improvement_earliest_tie",
    "domain_loss_weight": 1.0,
    "grl_progress_denominator": "planned_30_epoch_update_budget",
    "mmd_kernel_factors": [0.5, 1.0, 2.0],
    "mmd_estimator": "literal_empirical_mean_embedding_v_statistic",
    "mmd_bandwidth_pairs": "unique_upper_triangle_keep_off_diagonal_zeros",
    "mmd_kernel": "exp(-squared_distance/(2*bandwidth))",
    "mmd_bandwidth_gradient": "detached",
    "mmd_invalid_bandwidth": "stop",
    "mmd_feature_normalization": "l2_per_sample",
    "validation_batch_size": 64,
    "num_workers": 2,
}


def load_config(run_id: str) -> dict:
    if run_id not in RUNS:
        raise ValueError(f"Unknown run {run_id!r}; choose one of {tuple(RUNS)}")
    base = json.loads((CONFIG_DIR / "base.json").read_text())
    specific = json.loads((CONFIG_DIR / f"{run_id}.json").read_text())
    config = {**base, **specific}
    for key, expected in LOCKED_BASE.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Locked configuration field changed: {key}")
    method, strength = RUNS[run_id]
    if config.get("run_id") != run_id or config.get("method") != method:
        raise RuntimeError("Run-specific configuration identity mismatch")
    if config.get("mmd_lambda") != strength:
        raise RuntimeError("Run-specific MMD strength mismatch")
    return config
