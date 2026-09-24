"""Task 3 notebook block 06: run the approved DAN-DG lambda=0.1 study.

Paste this file's complete contents into one Colab cell. The block verifies the frozen
implementation and the completed lambda=1 evidence, persists the student's explicit
authorization, launches only lambda=0.1, audits its outputs, and then stops for review.
It never loads Sketch and never starts lambda=10 automatically.
"""

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


PROTOCOL_VERSION = "task3-approved-2026-09-24-v1"
EXPECTED_REPOSITORY_COMMIT = "19208b4c62acb980fb3246f30e062784b90d8dfc"
EXPECTED_CODE_TREE_SHA256 = (
    "4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44"
)
EXPECTED_CODE_PREFLIGHT_SHA256 = (
    "40ef37d7ae08ece5526e5588e446e5301158cc4bd444e2213cedfc0a9bf73eee"
)
EXPECTED_LAMBDA_1_CHECKPOINT_SHA256 = (
    "a44bff8e519134459c2d6bf056785801f9944945d160e516b4da7f5e2752992b"
)
EXPECTED_LAMBDA_1_HISTORY_SHA256 = (
    "321e267d9b628e357811da617fc5dac07149bdcbd30db030f4c8b65c20518ed7"
)
EXPECTED_LAMBDA_1_MANIFEST_SHA256 = (
    "33eb66823509417c758095c820efebdce2b8d90b64a3eecde64e3eaecea83e2f"
)

RUN_ID = "dan_dg_0p1"
EXPECTED_LAMBDA = 0.1

CODE_ROOT = Path("/content/atml_pa1_task3_source")
SOURCE_ROOT = Path("/content/task3_pacs_sources_v1")
ATML_DRIVE_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_DRIVE_ROOT = ATML_DRIVE_ROOT / "task3_domain_generalization_20260924"
PROVENANCE_ROOT = TASK3_DRIVE_ROOT / "provenance"
SOURCE_PROTOCOL = (
    TASK3_DRIVE_ROOT / "source_protocol" / "pacs_sources_seed6304.json"
)
TRAINING_ROOT = TASK3_DRIVE_ROOT / "training"
RUN_DIRECTORY = TRAINING_ROOT / RUN_ID

TASK2_ROOT = ATML_DRIVE_ROOT / "task2_corrected_normalized_v3_20260923"
COMMON_INITIALIZATION = (
    TASK2_ROOT / "initialization" / "resnet18_v1_seed6304_common.pt"
)
PREREGISTRATION = (
    CODE_ROOT / "task3" / "preregistration" / "DAN_DG_STRENGTH_EXPECTATION.md"
)
CODE_PREFLIGHT = PROVENANCE_ROOT / "code_preflight.json"
IMPLEMENTATION_RECORD = PROVENANCE_ROOT / "implementation_verification.json"
LAMBDA_1_COMPLETION = PROVENANCE_ROOT / "dan_dg_1_training_completion.json"
LAMBDA_1_DIAGNOSTIC = PROVENANCE_ROOT / "dan_dg_1_instability_diagnostic.json"
AUTHORIZATION_RECORD = PROVENANCE_ROOT / "dan_dg_0p1_authorization.json"
COMPLETION_RECORD = PROVENANCE_ROOT / "dan_dg_0p1_training_completion.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} is not a JSON object: {path}")
    return payload


def require_fields(record: dict, expected: dict, description: str) -> None:
    for name, value in expected.items():
        if record.get(name) != value:
            raise RuntimeError(
                f"{description} field {name!r} differs: "
                f"{record.get(name)!r} != {value!r}"
            )


def require_recorded_artifact(entry: dict, expected_sha256: str, name: str) -> Path:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise RuntimeError(f"The lambda=1 completion record has no valid {name} entry")
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"The recorded lambda=1 {name} is missing: {path}")
    actual = sha256_file(path)
    if entry.get("sha256") != expected_sha256 or actual != expected_sha256:
        raise RuntimeError(
            f"The lambda=1 {name} hash differs from the reviewed result"
        )
    return path


# Gate 1: exact frozen implementation and repository state.
implementation = load_json(IMPLEMENTATION_RECORD, "Block 04 implementation record")
require_fields(
    implementation,
    {
        "status": "TASK3_BLOCK_04_PASS",
        "protocol_version": PROTOCOL_VERSION,
        "source_only_phase": True,
        "training_started": False,
        "sketch_images_accessed": 0,
        "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
    },
    "Block 04 implementation record",
)
if implementation.get("unit_tests") != {"count": 14, "passed": 14, "failed": 0}:
    raise RuntimeError("Block 04 does not certify all 14 target-free tests")
if sha256_file(CODE_PREFLIGHT) != EXPECTED_CODE_PREFLIGHT_SHA256:
    raise RuntimeError("The code-preflight record changed after Block 04")
current_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
).strip()
if current_commit != EXPECTED_REPOSITORY_COMMIT:
    raise RuntimeError(
        "The Colab repository commit changed after the approved implementation gate: "
        f"{current_commit} != {EXPECTED_REPOSITORY_COMMIT}"
    )

# Gate 2: preserve and authenticate the reviewed lambda=1 main result.
lambda_1 = load_json(LAMBDA_1_COMPLETION, "lambda=1 completion record")
require_fields(
    lambda_1,
    {
        "status": "TASK3_BLOCK_05_DAN_DG_1_PASS",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": "dan_dg_1",
        "method": "dan_dg",
        "mmd_lambda": 1.0,
        "source_only_phase": True,
        "training_completed": True,
        "sketch_images_accessed": 0,
        "repository_commit": EXPECTED_REPOSITORY_COMMIT,
        "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
        "epochs_completed": 8,
        "best_epoch": 3,
        "best_mean_source_macro_f1": 0.8693286334613551,
    },
    "lambda=1 completion record",
)
require_recorded_artifact(
    lambda_1.get("best_checkpoint"),
    EXPECTED_LAMBDA_1_CHECKPOINT_SHA256,
    "selected checkpoint",
)
require_recorded_artifact(
    lambda_1.get("history"), EXPECTED_LAMBDA_1_HISTORY_SHA256, "history"
)
require_recorded_artifact(
    lambda_1.get("run_manifest"),
    EXPECTED_LAMBDA_1_MANIFEST_SHA256,
    "run manifest",
)

diagnostic = load_json(LAMBDA_1_DIAGNOSTIC, "reviewed lambda=1 diagnostic")
if diagnostic.get("source_only_phase", True) is not True:
    raise RuntimeError("The lambda=1 diagnostic does not preserve source-only status")
for key, value in diagnostic.items():
    if key.startswith("sketch_images_") and value != 0:
        raise RuntimeError(f"The lambda=1 diagnostic reports Sketch access in {key}")

# Gate 3: refuse overwrite, implicit resume, or an already completed study run.
if COMPLETION_RECORD.exists():
    raise FileExistsError(
        f"The lambda=0.1 completion record already exists: {COMPLETION_RECORD}"
    )
if RUN_DIRECTORY.exists():
    if (RUN_DIRECTORY / "run.json").is_file():
        raise FileExistsError("DAN-DG lambda=0.1 is already complete; do not rerun it")
    if (RUN_DIRECTORY / "last.pt").is_file():
        raise RuntimeError(
            "A completed-epoch lambda=0.1 checkpoint already exists. Do not silently "
            "resume it; preserve the directory and request a reviewed resume block."
        )
    raise RuntimeError(f"Unexpected lambda=0.1 run directory: {RUN_DIRECTORY}")

# Persist exactly what was approved before launching the run.
authorization = {
    "status": "TASK3_DAN_DG_0P1_AUTHORIZED",
    "protocol_version": PROTOCOL_VERSION,
    "student_approval_received": True,
    "approval_basis": "source_only_review_of_prescribed_dan_dg_lambda_1",
    "approved_next_run": RUN_ID,
    "approved_mmd_lambda": EXPECTED_LAMBDA,
    "all_other_settings_unchanged": True,
    "gradient_clipping_max_norm": 20.0,
    "lambda_1_main_result_immutable": True,
    "lambda_10_requires_lambda_0p1_completion_and_review": True,
    "modified_bandwidth_experiment_postponed_until_required_runs_complete": True,
    "source_only_phase": True,
    "sketch_images_accessed": 0,
    "repository_commit": current_commit,
    "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
    "lambda_1_completion_record": {
        "path": str(LAMBDA_1_COMPLETION),
        "sha256": sha256_file(LAMBDA_1_COMPLETION),
    },
    "lambda_1_diagnostic_record": {
        "path": str(LAMBDA_1_DIAGNOSTIC),
        "sha256": sha256_file(LAMBDA_1_DIAGNOSTIC),
    },
}
if AUTHORIZATION_RECORD.exists():
    if load_json(AUTHORIZATION_RECORD, "lambda=0.1 authorization") != authorization:
        raise RuntimeError("An existing lambda=0.1 authorization record differs")
else:
    atomic_write_json(authorization, AUTHORIZATION_RECORD)

command = [
    sys.executable,
    "-m",
    "task3.train",
    "--run-id",
    RUN_ID,
    "--pacs-source-root",
    str(SOURCE_ROOT),
    "--protocol",
    str(SOURCE_PROTOCOL),
    "--initialization",
    str(COMMON_INITIALIZATION),
    "--preregistration",
    str(PREREGISTRATION),
    "--code-preflight",
    str(CODE_PREFLIGHT),
    "--output",
    str(TRAINING_ROOT),
]

print("All Block 06 pre-training gates passed.")
print("Starting DAN-DG controlled-study run with lambda_DG = 0.1.")
print("Everything except lambda_DG is identical to the reviewed lambda=1 run.")
print("Updates per epoch: 235")
print("Maximum epochs: 30")
print("Early-stopping patience: 5")
print("Selection: unweighted mean source-validation macro-F1")
print("Gradient clipping max-norm: 20")
print("Sketch images accessible to training: 0")
print("No lambda=10 or SAM run will start from this block.\n")
subprocess.run(command, cwd=CODE_ROOT, check=True)

# Audit the completed lambda=0.1 artifacts.
run_manifest_path = RUN_DIRECTORY / "run.json"
history_path = RUN_DIRECTORY / "history.csv"
best_validation_path = RUN_DIRECTORY / "best_source_validation.json"
best_checkpoint_path = RUN_DIRECTORY / "best.pt"
run_manifest = load_json(run_manifest_path, "lambda=0.1 run manifest")
require_fields(
    run_manifest,
    {
        "status": "TASK3_RUN_COMPLETE",
        "run_id": RUN_ID,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "steps_per_source_epoch": 235,
    },
    "lambda=0.1 run manifest",
)
identity = run_manifest.get("identity")
if not isinstance(identity, dict):
    raise RuntimeError("The lambda=0.1 run manifest has no identity")
config = identity.get("config")
if not isinstance(config, dict):
    raise RuntimeError("The lambda=0.1 run identity has no configuration")
require_fields(
    config,
    {
        "run_id": RUN_ID,
        "method": "dan_dg",
        "mmd_lambda": EXPECTED_LAMBDA,
        "gradient_clipping": 20.0,
        "source_batch_per_domain": 8,
        "steps_per_source_epoch": 235,
        "epochs": 30,
        "patience": 5,
        "selection_metric": "unweighted_mean_source_validation_macro_f1",
        "mmd_feature_normalization": "l2_per_sample_mmd_input_only",
        "mmd_bandwidth_pairs": (
            "strict_upper_triangle_keep_off_diagonal_zeros"
        ),
        "mmd_kernel": "exp(-squared_distance/(2*bandwidth))",
        "mmd_estimator": "literal_empirical_mean_embedding_v_statistic",
    },
    "lambda=0.1 configuration",
)
require_fields(
    identity,
    {
        "code_sha256": EXPECTED_CODE_TREE_SHA256,
        "code_preflight_sha256": EXPECTED_CODE_PREFLIGHT_SHA256,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
    },
    "lambda=0.1 run identity",
)

epochs_completed = run_manifest.get("epochs_completed")
best_epoch = run_manifest.get("best_epoch")
best_f1 = run_manifest.get("best_mean_source_macro_f1")
if not isinstance(epochs_completed, int) or not 1 <= epochs_completed <= 30:
    raise RuntimeError("The lambda=0.1 run recorded an invalid epoch count")
if not isinstance(best_epoch, int) or not 1 <= best_epoch <= epochs_completed:
    raise RuntimeError("The lambda=0.1 run recorded an invalid best epoch")
if not isinstance(best_f1, (int, float)) or not math.isfinite(float(best_f1)):
    raise RuntimeError("The lambda=0.1 run recorded an invalid selected macro-F1")

best_validation = load_json(best_validation_path, "selected source validation")
if best_validation.get("epoch") != best_epoch:
    raise RuntimeError("Selected source-validation epoch differs from run.json")
if not math.isclose(
    float(best_validation.get("mean_source_macro_f1")),
    float(best_f1),
    rel_tol=0.0,
    abs_tol=1e-15,
):
    raise RuntimeError("Selected source macro-F1 differs from run.json")

if not best_checkpoint_path.is_file():
    raise FileNotFoundError("The selected lambda=0.1 checkpoint is missing")
best_checkpoint_sha256 = sha256_file(best_checkpoint_path)
if best_checkpoint_sha256 != run_manifest.get("best_checkpoint_sha256"):
    raise RuntimeError("The selected lambda=0.1 checkpoint hash differs from run.json")
if not history_path.is_file():
    raise FileNotFoundError("The lambda=0.1 history is missing")
with history_path.open(newline="") as handle:
    history = list(csv.DictReader(handle))
if len(history) != epochs_completed:
    raise RuntimeError("The lambda=0.1 history length differs from run.json")
if [int(row["epoch"]) for row in history] != list(range(1, epochs_completed + 1)):
    raise RuntimeError("The lambda=0.1 history epochs are not consecutive")

completion = {
    "status": "TASK3_BLOCK_06_DAN_DG_0P1_PASS",
    "protocol_version": PROTOCOL_VERSION,
    "run_id": RUN_ID,
    "method": "dan_dg",
    "mmd_lambda": EXPECTED_LAMBDA,
    "source_only_phase": True,
    "training_completed": True,
    "review_required_before_next_run": True,
    "next_run_started": False,
    "sketch_images_accessed": 0,
    "repository_commit": current_commit,
    "code_tree_sha256": identity["code_sha256"],
    "epochs_completed": epochs_completed,
    "best_epoch": best_epoch,
    "best_mean_source_macro_f1": float(best_f1),
    "best_source_validation": {
        key: value for key, value in best_validation.items() if key != "epoch"
    },
    "authorization": {
        "path": str(AUTHORIZATION_RECORD),
        "sha256": sha256_file(AUTHORIZATION_RECORD),
    },
    "best_checkpoint": {
        "path": str(best_checkpoint_path),
        "sha256": best_checkpoint_sha256,
    },
    "history": {
        "path": str(history_path),
        "sha256": sha256_file(history_path),
        "rows": len(history),
    },
    "run_manifest": {
        "path": str(run_manifest_path),
        "sha256": sha256_file(run_manifest_path),
    },
}
atomic_write_json(completion, COMPLETION_RECORD)

print("\nDAN-DG lambda=0.1 epoch summary:")
for row in history:
    print(
        f"epoch={int(row['epoch']):02d} "
        f"classification={float(row['classification_loss']):.6f} "
        f"mmd={float(row['mmd_loss']):.6f} "
        f"gradient={float(row['gradient_norm']):.6f} "
        f"clip_fraction={float(row['gradient_clipped_fraction']):.4f} "
        f"mean_f1={float(row['mean_source_macro_f1']):.6f} "
        f"worst_f1={float(row['worst_source_macro_f1']):.6f}"
    )

print("\nAudited completion record:\n")
print(json.dumps(completion, indent=2))
print(f"\nSaved at: {COMPLETION_RECORD}")
print("Sketch images accessed: 0")
print("No subsequent run was started.")
print("Send the complete final output for review before running lambda=10.")
