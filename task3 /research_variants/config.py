"""Locked configuration for the approved DAN-DG bandwidth-floor study."""

from __future__ import annotations

import json
from pathlib import Path

from task3.config import CONFIG_DIR, LOCKED_BASE, sha256_file


VARIANT_ID = "dan_dg_initial_bandwidth_floor_v1"
VARIANT_PROTOCOL_VERSION = "task3-research-bandwidth-floor-2026-09-25-v1"

VARIANT_RUNS = {
    "dan_dg_floor_0p1": 0.1,
    "dan_dg_floor_1": 1.0,
    "dan_dg_floor_10": 10.0,
}

RESEARCH_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "preregistration"
    / "DAN_DG_BANDWIDTH_FLOOR_STUDY.md"
)

VARIANT_LOCKED = {
    "research_variant": True,
    "variant_id": VARIANT_ID,
    "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
    "primary_protocol_replacement": False,
    "bandwidth_policy": "max_current_batch_median_and_pair_initialization_floor",
    "floor_scope": "one_frozen_value_per_unordered_source_pair",
    "floor_calibration_model": "common_initialization_before_any_training_update",
    "floor_calibration_data": "source_training_records_only",
    "floor_calibration_labels_used": False,
    "floor_calibration_transform": "locked_deterministic_validation_center_crop",
    "floor_calibration_steps": 235,
    "floor_calibration_batch_per_domain": 8,
    "floor_reduction": "median_of_235_current_batch_pair_medians",
    "floor_multiplier": 1.0,
    "floor_reused_across_lambdas": True,
    "target_domain_used": False,
}


def verify_research_protocol(path: Path = RESEARCH_PROTOCOL_PATH) -> str:
    if path.resolve() != RESEARCH_PROTOCOL_PATH.resolve():
        raise RuntimeError("Unexpected bandwidth-floor research protocol path")
    text = path.read_text()
    required = (
        VARIANT_ID,
        VARIANT_PROTOCOL_VERSION,
        "supplementary",
        "Sketch",
        "median of the 235",
        "primary",
    )
    if any(value not in text for value in required):
        raise RuntimeError("Bandwidth-floor research protocol is incomplete")
    return sha256_file(path)


def load_variant_config(run_id: str) -> dict:
    if run_id not in VARIANT_RUNS:
        raise ValueError(
            f"Unknown bandwidth-floor run {run_id!r}; "
            f"choose one of {tuple(VARIANT_RUNS)}"
        )

    base = json.loads((CONFIG_DIR / "base.json").read_text())
    specific = json.loads((CONFIG_DIR / f"{run_id}.json").read_text())
    config = {**base, **specific}

    for name, expected in LOCKED_BASE.items():
        if config.get(name) != expected:
            raise RuntimeError(f"Primary Task 3 invariant changed: {name}")
    for name, expected in VARIANT_LOCKED.items():
        if config.get(name) != expected:
            raise RuntimeError(f"Bandwidth-floor invariant changed: {name}")

    expected_identity = {
        "run_id": run_id,
        "method": "dan_dg_bandwidth_floor",
        "mmd_lambda": VARIANT_RUNS[run_id],
        "sam_rho": None,
    }
    for name, expected in expected_identity.items():
        if config.get(name) != expected:
            raise RuntimeError(
                f"Bandwidth-floor run identity changed: {name}="
                f"{config.get(name)!r}, expected {expected!r}"
            )
    return config

