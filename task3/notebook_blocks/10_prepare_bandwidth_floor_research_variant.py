"""Block 10: install, verify, authorize, and calibrate the research variant.

Run only after committing the complete approved research-variant file set. This block
does not train a model. It authenticates the original source-only evidence, verifies
the exact Git change set and all 25 target-free tests, writes the explicit authorization
record, and derives the three frozen initialization bandwidth floors without Sketch.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DIAGNOSTIC_COMMIT = "10e3c1e80a267ffe77456814071fa7895ac2743d"
EXPECTED_CODE_TREE_SHA256 = (
    "34d2778e76a5a7522a9c2eac2682eb4d2a09ee9de0e88b6d04d7d0a68e26ce13"
)
EXPECTED_RESEARCH_PROTOCOL_SHA256 = (
    "00c5233ff8925c553b631c8c14e25d52af7534c89ab3437af33098b3f769ad84"
)
VARIANT_ID = "dan_dg_initial_bandwidth_floor_v1"
VARIANT_PROTOCOL_VERSION = "task3-research-bandwidth-floor-2026-09-25-v1"

REQUIRED_UPDATE = {
    "task3/README.md",
    "task3/calibrate_bandwidth_floor.py",
    "task3/configs/dan_dg_floor_0p1.json",
    "task3/configs/dan_dg_floor_1.json",
    "task3/configs/dan_dg_floor_10.json",
    "task3/docs/DAN_DG_BANDWIDTH_FLOOR_RESEARCH_VARIANT.md",
    "task3/notebook_blocks/10_prepare_bandwidth_floor_research_variant.py",
    "task3/notebook_blocks/11_train_dan_dg_floor_1.py",
    "task3/notebook_blocks/12_train_dan_dg_floor_0p1.py",
    "task3/notebook_blocks/13_train_dan_dg_floor_10.py",
    "task3/preregistration/DAN_DG_BANDWIDTH_FLOOR_STUDY.md",
    "task3/provenance/RUN_LOG.md",
    "task3/research_variants/__init__.py",
    "task3/research_variants/bandwidth_floor.py",
    "task3/research_variants/config.py",
    "task3/research_variants/notebook_runner.py",
    "task3/tests/test_task3_bandwidth_floor.py",
    "task3/train_bandwidth_floor.py",
}

CODE_ROOT = Path("/content/atml_pa1_task3_source")
SOURCE_ROOT = Path("/content/task3_pacs_sources_v1")
ATML_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_ROOT = ATML_ROOT / "task3_domain_generalization_20260924"
PRIMARY_PROVENANCE = TASK3_ROOT / "provenance"
SOURCE_PROTOCOL = TASK3_ROOT / "source_protocol" / "pacs_sources_seed6304.json"
COMMON_INITIALIZATION = (
    ATML_ROOT
    / "task2_corrected_normalized_v3_20260923"
    / "initialization"
    / "resnet18_v1_seed6304_common.pt"
)
SOURCE_DIAGNOSTICS_COMPLETION = (
    PRIMARY_PROVENANCE / "source_diagnostics_completion.json"
)
VARIANT_ROOT = TASK3_ROOT / "research_variants" / VARIANT_ID
VARIANT_PROVENANCE = VARIANT_ROOT / "provenance"
AUTHORIZATION = VARIANT_PROVENANCE / "variant_authorization.json"
CALIBRATION = VARIANT_ROOT / "calibration" / "bandwidth_floors.json"
CALIBRATION_COMPLETION = VARIANT_PROVENANCE / "calibration_completion.json"
RESEARCH_PROTOCOL = (
    CODE_ROOT / "task3" / "preregistration" / "DAN_DG_BANDWIDTH_FLOOR_STUDY.md"
)

ORIGINAL_HISTORIES = {
    "dan_dg_0p1": (
        TASK3_ROOT / "training" / "dan_dg_0p1" / "history.csv",
        "fc92efd81ea529a48373977f9d4973e754eb53cfca7622c8513ad5cfc64b35b8",
    ),
    "dan_dg_1": (
        TASK3_ROOT / "training" / "dan_dg_1" / "history.csv",
        "321e267d9b628e357811da617fc5dac07149bdcbd30db030f4c8b65c20518ed7",
    ),
    "dan_dg_10": (
        TASK3_ROOT / "training" / "dan_dg_10" / "history.csv",
        "dec8773f2a0642b2aecea85619dbd8bdc19a74723c06a5ce9ca24ce9c8b4b918",
    ),
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


def verify_file(path: Path, expected: str, description: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"{description} identity differs: {path}")


if not (CODE_ROOT / ".git").is_dir():
    raise FileNotFoundError(f"Colab repository checkout is missing: {CODE_ROOT}")
dirty = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True
).strip()
if dirty:
    raise RuntimeError("The Colab checkout has uncommitted changes:\n" + dirty)
subprocess.run(["git", "pull", "--ff-only"], cwd=CODE_ROOT, check=True)
research_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
).strip()
ancestor = subprocess.run(
    ["git", "merge-base", "--is-ancestor", DIAGNOSTIC_COMMIT, research_commit],
    cwd=CODE_ROOT,
)
if ancestor.returncode != 0 or research_commit == DIAGNOSTIC_COMMIT:
    raise RuntimeError("Research commit is not a strict diagnostic-commit descendant")
changed = set(
    filter(
        None,
        subprocess.check_output(
            ["git", "diff", "--name-only", f"{DIAGNOSTIC_COMMIT}..{research_commit}"],
            cwd=CODE_ROOT,
            text=True,
        ).splitlines(),
    )
)
missing = REQUIRED_UPDATE - changed
unexpected = changed - REQUIRED_UPDATE
if missing or unexpected:
    raise RuntimeError(
        "Research Git update differs from the approved file set. "
        f"Missing={sorted(missing)}, unexpected={sorted(unexpected)}"
    )

sys.path.insert(0, str(CODE_ROOT))
from task3.config import code_tree_sha256  # noqa: E402


actual_code_sha256 = code_tree_sha256(CODE_ROOT)
if actual_code_sha256 != EXPECTED_CODE_TREE_SHA256:
    raise RuntimeError(
        f"Research code tree differs: {actual_code_sha256} "
        f"!= {EXPECTED_CODE_TREE_SHA256}"
    )
verify_file(
    RESEARCH_PROTOCOL,
    EXPECTED_RESEARCH_PROTOCOL_SHA256,
    "research preregistration",
)
test_result = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "task3/tests", "-v"],
    cwd=CODE_ROOT,
)
if test_result.returncode != 0:
    raise RuntimeError("The 25-test target-free suite failed")

diagnostics = load_json(
    SOURCE_DIAGNOSTICS_COMPLETION, "Block 09 source diagnostics completion"
)
expected_diagnostics = {
    "status": "TASK3_BLOCK_09_SOURCE_DIAGNOSTICS_PASS",
    "source_only_phase": True,
    "diagnostics_completed": True,
    "experiment_lock_created": False,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "diagnostic_commit": DIAGNOSTIC_COMMIT,
    "diagnostic_code_tree_sha256": (
        "df3de3b753a6008ed8ccae50d4ef8f278165dcc1bbdfe57e9acb005ce08f1989"
    ),
}
for name, value in expected_diagnostics.items():
    if diagnostics.get(name) != value:
        raise RuntimeError(f"Block 09 completion field differs: {name}")
if diagnostics.get("target_free_tests") != {"count": 18, "passed": 18, "failed": 0}:
    raise RuntimeError("Block 09 did not certify 18 target-free tests")
result_entry = diagnostics.get("result", {})
verify_file(
    Path(result_entry.get("path", "")),
    "cdf83aa3fb41128f39a89c3358eeb7a7b09288862c98274473bb7aeb41830e1f",
    "source diagnostic result",
)

history_identities = {}
for run_id, (path, expected) in ORIGINAL_HISTORIES.items():
    verify_file(path, expected, f"original {run_id} history")
    history_identities[run_id] = {"path": str(path), "sha256": expected}

if CALIBRATION.exists() or CALIBRATION_COMPLETION.exists():
    raise FileExistsError("Research calibration already exists; do not overwrite it")

authorization = {
    "status": "TASK3_BANDWIDTH_FLOOR_VARIANT_AUTHORIZED",
    "variant_id": VARIANT_ID,
    "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
    "approval_date": "2026-09-25",
    "research_variant": True,
    "primary_protocol_replacement": False,
    "single_controlled_change": (
        "For each source pair, replace the current median used for bandwidths with "
        "max(current median, frozen initialization-derived pair floor)."
    ),
    "approved_lambdas": [0.1, 1.0, 10.0],
    "run_order": ["dan_dg_floor_1", "dan_dg_floor_0p1", "dan_dg_floor_10"],
    "floor_calibration": {
        "model": "authenticated common initialization before any update",
        "data": "source training records only",
        "labels_used_in_computation": False,
        "transform": "locked deterministic validation center crop",
        "steps": 235,
        "batch_per_domain": 8,
        "reduction": "median of the 235 current-batch medians per pair",
        "multiplier": 1.0,
        "reused_unchanged_across_lambdas": True,
    },
    "unchanged_primary_settings": [
        "source split and seed",
        "source-only embargo",
        "model and common initialization",
        "training transforms and source sampler",
        "eight examples per domain and 235 updates per epoch",
        "feature normalization outside the bandwidth floor",
        "kernel factors, RBF convention, V-statistic, and pair reduction",
        "AdamW settings",
        "gradient clipping max norm 20",
        "BatchNorm policy",
        "30-epoch budget and patience 5",
        "source-validation selection metric",
    ],
    "motivation": {
        "prescribed_lambda_1_best_mean_source_macro_f1": 0.8693286334613551,
        "erm_mean_source_macro_f1": 0.9426262342459099,
        "lambda_0p1_best_mean_source_macro_f1": 0.9462297763360951,
        "lambda_1_epoch_1_gradient_norm": 156.746653,
        "lambda_1_epoch_8_gradient_norm": 782.310502,
        "lambda_1_epoch_1_clipped_updates": 224,
        "lambda_1_later_clipped_updates_per_epoch": 235,
        "lambda_1_epoch_1_pair_median_range": [0.026499499, 0.030206157],
        "lambda_1_epoch_8_pair_median_range": [0.000040634576, 0.000041435890],
        "lambda_1_delta_sharp": 94.29026794433594,
        "erm_delta_sharp": 0.24889972805976868,
        "lambda_0p1_total_clipped_updates": 70,
        "lambda_0p1_total_updates": 2350,
    },
    "source_only_phase": True,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "repository_commit": research_commit,
    "code_tree_sha256": actual_code_sha256,
    "source_protocol_sha256": (
        "626d8517b44ad50c0219adf49e827de6538561386791bed29a9153a589cd6abc"
    ),
    "source_snapshot_sha256": (
        "8ded350769ee15739f8420e755e50ff4377068a4f54ab1c0ba39d5b125e658d2"
    ),
    "initialization_state_sha256": (
        "4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3"
    ),
    "research_protocol": {
        "path": str(RESEARCH_PROTOCOL),
        "sha256": EXPECTED_RESEARCH_PROTOCOL_SHA256,
    },
    "source_diagnostics_completion": {
        "path": str(SOURCE_DIAGNOSTICS_COMPLETION),
        "sha256": sha256_file(SOURCE_DIAGNOSTICS_COMPLETION),
    },
    "original_histories": history_identities,
}
atomic_write_json(authorization, AUTHORIZATION)

command = [
    sys.executable,
    "-m",
    "task3.calibrate_bandwidth_floor",
    "--code-root",
    str(CODE_ROOT),
    "--pacs-source-root",
    str(SOURCE_ROOT),
    "--protocol",
    str(SOURCE_PROTOCOL),
    "--initialization",
    str(COMMON_INITIALIZATION),
    "--research-protocol",
    str(RESEARCH_PROTOCOL),
    "--authorization",
    str(AUTHORIZATION),
    "--output",
    str(CALIBRATION),
    "--num-workers",
    "2",
]
print("All Block 10 pre-calibration gates passed.")
print("Repository commit:", research_commit)
print("Target-free tests passed: 25/25")
print("Calibration: common initialization, 235 source-only center-crop batches")
print("Sketch images accessible to calibration: 0")
process = subprocess.run(command, cwd=CODE_ROOT)
if process.returncode != 0:
    raise RuntimeError(f"Bandwidth-floor calibration failed: {process.returncode}")

calibration = load_json(CALIBRATION, "bandwidth-floor calibration")
expected_calibration = {
    "status": "TASK3_BANDWIDTH_FLOOR_CALIBRATION_PASS",
    "variant_id": VARIANT_ID,
    "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
    "research_variant": True,
    "primary_protocol_replacement": False,
    "source_only_phase": True,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "class_labels_used_in_calibration_computation": False,
}
for name, value in expected_calibration.items():
    if calibration.get(name) != value:
        raise RuntimeError(f"Calibration output field differs: {name}")
pair_floors = {}
for pair in ("photo__art_painting", "photo__cartoon", "art_painting__cartoon"):
    details = calibration.get("pair_floors", {}).get(pair, {})
    floor = float(details.get("floor_squared_distance", 0.0))
    if floor <= 0 or details.get("sample_count") != 235:
        raise RuntimeError(f"Invalid calibrated floor for {pair}")
    if len(calibration.get("pair_batch_medians", {}).get(pair, [])) != 235:
        raise RuntimeError(f"Incomplete calibration trace for {pair}")
    pair_floors[pair] = floor

completion = {
    "status": "TASK3_BLOCK_10_BANDWIDTH_FLOOR_CALIBRATION_PASS",
    "variant_id": VARIANT_ID,
    "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
    "research_variant": True,
    "primary_protocol_replacement": False,
    "source_only_phase": True,
    "calibration_completed": True,
    "training_started": False,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "repository_commit": research_commit,
    "code_tree_sha256": actual_code_sha256,
    "target_free_tests_passed": 25,
    "authorization": {"path": str(AUTHORIZATION), "sha256": sha256_file(AUTHORIZATION)},
    "calibration": {
        "path": str(CALIBRATION),
        "sha256": sha256_file(CALIBRATION),
        "pair_floors": pair_floors,
    },
}
atomic_write_json(completion, CALIBRATION_COMPLETION)
print("\nAudited Block 10 completion:\n", json.dumps(completion, indent=2))
print("No training run was started.")
print("Sketch images accessed: 0")
