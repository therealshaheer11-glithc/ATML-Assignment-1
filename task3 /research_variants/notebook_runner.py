"""Colab-side audited launcher for one bandwidth-floor research run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from task3.config import code_tree_sha256
from task3.research_variants.bandwidth_floor import PAIR_NAMES
from task3.research_variants.config import (
    VARIANT_ID,
    VARIANT_PROTOCOL_VERSION,
    VARIANT_RUNS,
)


CODE_ROOT = Path("/content/atml_pa1_task3_source")
SOURCE_ROOT = Path("/content/task3_pacs_sources_v1")
ATML_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_ROOT = ATML_ROOT / "task3_domain_generalization_20260924"
VARIANT_ROOT = TASK3_ROOT / "research_variants" / VARIANT_ID
VARIANT_PROVENANCE = VARIANT_ROOT / "provenance"
VARIANT_TRAINING = VARIANT_ROOT / "training"
SOURCE_PROTOCOL = TASK3_ROOT / "source_protocol" / "pacs_sources_seed6304.json"
COMMON_INITIALIZATION = (
    ATML_ROOT
    / "task2_corrected_normalized_v3_20260923"
    / "initialization"
    / "resnet18_v1_seed6304_common.pt"
)
RESEARCH_PROTOCOL = (
    CODE_ROOT / "task3" / "preregistration" / "DAN_DG_BANDWIDTH_FLOOR_STUDY.md"
)
CALIBRATION = VARIANT_ROOT / "calibration" / "bandwidth_floors.json"
CALIBRATION_COMPLETION = VARIANT_PROVENANCE / "calibration_completion.json"

RUN_ORDER = ("dan_dg_floor_1", "dan_dg_floor_0p1", "dan_dg_floor_10")
RUN_BLOCKS = {
    "dan_dg_floor_1": (11, "TASK3_BLOCK_11_DAN_DG_FLOOR_1_PASS"),
    "dan_dg_floor_0p1": (12, "TASK3_BLOCK_12_DAN_DG_FLOOR_0P1_PASS"),
    "dan_dg_floor_10": (13, "TASK3_BLOCK_13_DAN_DG_FLOOR_10_PASS"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object")
    return value


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def verify_artifact(path: Path, expected_sha256: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{description} hash differs")
    return actual


def verify_calibration_gate() -> tuple[dict, dict, str, str]:
    completion = load_json(CALIBRATION_COMPLETION, "calibration completion")
    expected = {
        "status": "TASK3_BLOCK_10_BANDWIDTH_FLOOR_CALIBRATION_PASS",
        "variant_id": VARIANT_ID,
        "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
        "source_only_phase": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
        "target_free_tests_passed": 25,
    }
    for name, value in expected.items():
        if completion.get(name) != value:
            raise RuntimeError(f"Calibration completion field differs: {name}")

    calibration_entry = completion.get("calibration", {})
    calibration_sha256 = verify_artifact(
        Path(calibration_entry.get("path", "")),
        calibration_entry.get("sha256", ""),
        "frozen calibration",
    )
    calibration = load_json(CALIBRATION, "frozen calibration")
    if calibration.get("sketch_images_accessed") != 0:
        raise RuntimeError("Calibration reports Sketch access")
    if calibration.get("class_labels_used_in_calibration_computation") is not False:
        raise RuntimeError("Calibration does not certify label exclusion")

    commit = completion.get("repository_commit")
    code_sha256 = completion.get("code_tree_sha256")
    if not isinstance(commit, str) or not isinstance(code_sha256, str):
        raise RuntimeError("Calibration completion lacks code identities")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
    ).strip()
    if current_commit != commit:
        raise RuntimeError("Repository changed after floor calibration")
    if code_tree_sha256(CODE_ROOT) != code_sha256:
        raise RuntimeError("Executable code changed after floor calibration")
    return completion, calibration, calibration_sha256, code_sha256


def verify_prerequisites(run_id: str) -> None:
    index = RUN_ORDER.index(run_id)
    for prior_run in RUN_ORDER[:index]:
        block_number, status = RUN_BLOCKS[prior_run]
        path = VARIANT_PROVENANCE / f"{prior_run}_training_completion.json"
        record = load_json(path, f"{prior_run} completion")
        expected = {
            "status": status,
            "run_id": prior_run,
            "mmd_lambda": VARIANT_RUNS[prior_run],
            "source_only_phase": True,
            "training_completed": True,
            "next_run_started": False,
            "sketch_images_accessed": 0,
        }
        for name, value in expected.items():
            if record.get(name) != value:
                raise RuntimeError(
                    f"Block {block_number} prerequisite differs: {name}"
                )
        for artifact_name in ("best_checkpoint", "history", "run_manifest"):
            artifact = record.get(artifact_name, {})
            verify_artifact(
                Path(artifact.get("path", "")),
                artifact.get("sha256", ""),
                f"{prior_run} {artifact_name}",
            )


def audit_history(path: Path, floors: dict[str, float]) -> tuple[list[dict], dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) > 30:
        raise RuntimeError("Variant history has an invalid epoch count")

    aggregate = {
        "maximum_gradient_norm": 0.0,
        "mean_clipped_fraction": 0.0,
        "mean_floor_activation_fraction": 0.0,
    }
    for row in rows:
        numeric_fields = (
            "classification_loss",
            "mmd_loss",
            "gradient_norm",
            "gradient_norm_after_clipping",
            "gradient_clipped_fraction",
            "mmd_floor_activation_fraction",
            "mean_source_macro_f1",
            "worst_source_macro_f1",
        )
        if any(not math.isfinite(float(row[name])) for name in numeric_fields):
            raise RuntimeError("Variant history contains non-finite diagnostics")
        if float(row["gradient_norm_after_clipping"]) > 20.0001:
            raise RuntimeError("Post-clip gradient exceeded the approved bound")
        for pair in PAIR_NAMES:
            recorded_floor = float(row[f"mmd_frozen_floor_{pair}"])
            if abs(recorded_floor - floors[pair]) > 1e-12:
                raise RuntimeError(f"Frozen floor changed for {pair}")
            current = float(row[f"mmd_current_median_{pair}"])
            effective = float(row[f"mmd_effective_median_{pair}"])
            activation = float(row[f"mmd_floor_activation_fraction_{pair}"])
            if current <= 0 or effective <= 0 or not 0 <= activation <= 1:
                raise RuntimeError(f"Invalid bandwidth diagnostics for {pair}")
            if effective + 1e-12 < max(current, floors[pair]):
                raise RuntimeError(f"Effective median violates the floor for {pair}")
        aggregate["maximum_gradient_norm"] = max(
            aggregate["maximum_gradient_norm"], float(row["gradient_norm"])
        )
        aggregate["mean_clipped_fraction"] += float(
            row["gradient_clipped_fraction"]
        )
        aggregate["mean_floor_activation_fraction"] += float(
            row["mmd_floor_activation_fraction"]
        )

    aggregate["mean_clipped_fraction"] /= len(rows)
    aggregate["mean_floor_activation_fraction"] /= len(rows)
    return rows, aggregate


def run_variant_block(run_id: str) -> dict:
    if run_id not in VARIANT_RUNS:
        raise ValueError(f"Unknown research run: {run_id}")
    block_number, status = RUN_BLOCKS[run_id]
    completion_path = VARIANT_PROVENANCE / f"{run_id}_training_completion.json"
    run_dir = VARIANT_TRAINING / run_id
    if completion_path.exists() or run_dir.exists():
        raise FileExistsError(f"{run_id} already has outputs; do not overwrite them")

    completion, calibration, calibration_sha256, code_sha256 = (
        verify_calibration_gate()
    )
    verify_prerequisites(run_id)
    floors = {
        pair: float(calibration["pair_floors"][pair]["floor_squared_distance"])
        for pair in PAIR_NAMES
    }

    print(f"All Block {block_number:02d} pre-training gates passed.")
    print("Research variant:", VARIANT_ID)
    print("Starting:", run_id, "lambda_DG =", VARIANT_RUNS[run_id])
    print("Frozen pair floors:", json.dumps(floors, indent=2))
    print("Only the effective bandwidth median is changed from the primary run.")
    print("Sketch images accessible to training: 0")

    command = [
        sys.executable,
        "-m",
        "task3.train_bandwidth_floor",
        "--run-id",
        run_id,
        "--pacs-source-root",
        str(SOURCE_ROOT),
        "--protocol",
        str(SOURCE_PROTOCOL),
        "--initialization",
        str(COMMON_INITIALIZATION),
        "--research-protocol",
        str(RESEARCH_PROTOCOL),
        "--calibration",
        str(CALIBRATION),
        "--output",
        str(VARIANT_TRAINING),
        "--device",
        "cuda",
        "--num-workers",
        "2",
    ]
    process = subprocess.run(command, cwd=CODE_ROOT)
    if process.returncode != 0:
        raise RuntimeError(f"{run_id} training failed with code {process.returncode}")

    manifest_path = run_dir / "run.json"
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best.pt"
    manifest = load_json(manifest_path, f"{run_id} manifest")
    expected_manifest = {
        "status": "TASK3_BANDWIDTH_FLOOR_RUN_COMPLETE",
        "run_id": run_id,
        "method": "dan_dg_bandwidth_floor",
        "mmd_lambda": VARIANT_RUNS[run_id],
        "calibration_sha256": calibration_sha256,
        "research_variant": True,
        "primary_protocol_replacement": False,
        "source_only_phase": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
    }
    for name, value in expected_manifest.items():
        if manifest.get(name) != value:
            raise RuntimeError(f"{run_id} manifest field differs: {name}")
    if manifest.get("identity", {}).get("code_sha256") != code_sha256:
        raise RuntimeError("Variant manifest code identity differs")
    if manifest.get("identity", {}).get("frozen_pair_floors") != floors:
        raise RuntimeError("Variant manifest floor identity differs")

    checkpoint_sha256 = verify_artifact(
        checkpoint_path,
        manifest.get("best_checkpoint_sha256", ""),
        f"{run_id} best checkpoint",
    )
    rows, aggregate = audit_history(history_path, floors)
    if len(rows) != int(manifest["epochs_completed"]):
        raise RuntimeError("Variant history length differs from manifest")
    best_record = load_json(
        run_dir / "best_source_validation.json", f"{run_id} best source metrics"
    )
    if int(best_record["epoch"]) != int(manifest["best_epoch"]):
        raise RuntimeError("Variant best epoch differs")
    if abs(
        float(best_record["mean_source_macro_f1"])
        - float(manifest["best_mean_source_macro_f1"])
    ) > 1e-12:
        raise RuntimeError("Variant best source F1 differs")

    result = {
        "status": status,
        "variant_id": VARIANT_ID,
        "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
        "run_id": run_id,
        "method": "dan_dg_bandwidth_floor",
        "mmd_lambda": VARIANT_RUNS[run_id],
        "research_variant": True,
        "primary_protocol_replacement": False,
        "source_only_phase": True,
        "training_completed": True,
        "review_required_before_next_run": True,
        "next_run_started": False,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
        "repository_commit": completion["repository_commit"],
        "code_tree_sha256": code_sha256,
        "calibration": {
            "path": str(CALIBRATION),
            "sha256": calibration_sha256,
            "pair_floors": floors,
        },
        "epochs_completed": len(rows),
        "best_epoch": int(manifest["best_epoch"]),
        "best_mean_source_macro_f1": float(
            manifest["best_mean_source_macro_f1"]
        ),
        "best_source_validation": best_record,
        "training_diagnostics": aggregate,
        "best_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "history": {
            "path": str(history_path),
            "sha256": sha256_file(history_path),
            "rows": len(rows),
        },
        "run_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
    }
    atomic_write_json(result, completion_path)

    print("\nEpoch summary:")
    for row in rows:
        print(
            f"epoch={int(row['epoch']):02d} "
            f"classification={float(row['classification_loss']):.6f} "
            f"mmd={float(row['mmd_loss']):.6f} "
            f"gradient={float(row['gradient_norm']):.6f} "
            f"clip_fraction={float(row['gradient_clipped_fraction']):.4f} "
            f"floor_active={float(row['mmd_floor_activation_fraction']):.4f} "
            f"mean_f1={float(row['mean_source_macro_f1']):.6f} "
            f"worst_f1={float(row['worst_source_macro_f1']):.6f}"
        )
    print("\nAudited completion record:\n", json.dumps(result, indent=2))
    print("Saved at:", completion_path)
    print("Sketch images accessed: 0")
    print("No subsequent run was started.")
    return result
