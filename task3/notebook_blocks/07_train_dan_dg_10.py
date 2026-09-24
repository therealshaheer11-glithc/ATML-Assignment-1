"""Task 3 notebook block 07: run the approved DAN-DG lambda=10 study.

Paste this file's complete contents into one Colab cell. The block authenticates the
frozen implementation and reviewed lambda=0.1 result, records that the review condition
for lambda=10 was satisfied, launches only lambda=10, audits successful outputs, and
then stops for review. It never loads Sketch and never starts SAM automatically.
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
EXPECTED_0P1_AUTHORIZATION_SHA256 = (
    "115e50f674a75660d135d9b4d9897879b15538d83ae85521f58c9206ff1e027c"
)
EXPECTED_0P1_CHECKPOINT_SHA256 = (
    "dc6036a28e3af8c281b143adb6f47b4d03676fa44c117e912ef9bcd959d8ca27"
)
EXPECTED_0P1_HISTORY_SHA256 = (
    "fc92efd81ea529a48373977f9d4973e754eb53cfca7622c8513ad5cfc64b35b8"
)
EXPECTED_0P1_MANIFEST_SHA256 = (
    "731f07bded36f6034e014eb217dca2b1b3a9c7836ed0e5bd72d0c2be3537672a"
)

RUN_ID = "dan_dg_10"
EXPECTED_LAMBDA = 10.0

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
LAMBDA_0P1_COMPLETION = PROVENANCE_ROOT / "dan_dg_0p1_training_completion.json"
AUTHORIZATION_RECORD = PROVENANCE_ROOT / "dan_dg_10_authorization.json"
COMPLETION_RECORD = PROVENANCE_ROOT / "dan_dg_10_training_completion.json"


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
        raise RuntimeError(f"The lambda=0.1 record has no valid {name} entry")
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"The recorded lambda=0.1 {name} is missing: {path}")
    actual = sha256_file(path)
    if entry.get("sha256") != expected_sha256 or actual != expected_sha256:
        raise RuntimeError(f"The lambda=0.1 {name} hash differs from the reviewed run")
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

# Gate 2: authenticate the completed and reviewed lambda=0.1 prerequisite.
lambda_0p1 = load_json(LAMBDA_0P1_COMPLETION, "lambda=0.1 completion record")
require_fields(
    lambda_0p1,
    {
        "status": "TASK3_BLOCK_06_DAN_DG_0P1_PASS",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": "dan_dg_0p1",
        "method": "dan_dg",
        "mmd_lambda": 0.1,
        "source_only_phase": True,
        "training_completed": True,
        "review_required_before_next_run": True,
        "next_run_started": False,
        "sketch_images_accessed": 0,
        "repository_commit": EXPECTED_REPOSITORY_COMMIT,
        "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
        "epochs_completed": 10,
        "best_epoch": 5,
        "best_mean_source_macro_f1": 0.9462297763360951,
    },
    "lambda=0.1 completion record",
)
require_recorded_artifact(
    lambda_0p1.get("authorization"),
    EXPECTED_0P1_AUTHORIZATION_SHA256,
    "authorization",
)
require_recorded_artifact(
    lambda_0p1.get("best_checkpoint"),
    EXPECTED_0P1_CHECKPOINT_SHA256,
    "selected checkpoint",
)
require_recorded_artifact(
    lambda_0p1.get("history"), EXPECTED_0P1_HISTORY_SHA256, "history"
)
require_recorded_artifact(
    lambda_0p1.get("run_manifest"),
    EXPECTED_0P1_MANIFEST_SHA256,
    "run manifest",
)

# Gate 3: refuse overwrite, implicit resume, or an already completed lambda=10 run.
if COMPLETION_RECORD.exists():
    raise FileExistsError(
        f"The lambda=10 completion record already exists: {COMPLETION_RECORD}"
    )
if RUN_DIRECTORY.exists():
    if (RUN_DIRECTORY / "run.json").is_file():
        raise FileExistsError("DAN-DG lambda=10 is already complete; do not rerun it")
    if (RUN_DIRECTORY / "last.pt").is_file():
        raise RuntimeError(
            "A completed-epoch lambda=10 checkpoint already exists. Do not silently "
            "resume it; preserve the directory and request a reviewed resume block."
        )
    raise RuntimeError(f"Unexpected lambda=10 run directory: {RUN_DIRECTORY}")

# Persist the already-approved decision and the completed lambda=0.1 review gate.
authorization = {
    "status": "TASK3_DAN_DG_10_AUTHORIZED_AFTER_0P1_REVIEW",
    "protocol_version": PROTOCOL_VERSION,
    "student_approval_received": True,
    "lambda_0p1_completed_and_reviewed": True,
    "review_basis": "source_only_metrics_and_optimization_diagnostics",
    "approved_next_run": RUN_ID,
    "approved_mmd_lambda": EXPECTED_LAMBDA,
    "all_other_settings_unchanged": True,
    "gradient_clipping_max_norm": 20.0,
    "strong_alignment_instability_expected_and_preserved_if_observed": True,
    "lambda_1_main_result_immutable": True,
    "modified_bandwidth_experiment_postponed_until_required_runs_complete": True,
    "source_only_phase": True,
    "sketch_images_accessed": 0,
    "repository_commit": current_commit,
    "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
    "lambda_0p1_completion_record": {
        "path": str(LAMBDA_0P1_COMPLETION),
        "sha256": sha256_file(LAMBDA_0P1_COMPLETION),
    },
}
if AUTHORIZATION_RECORD.exists():
    if load_json(AUTHORIZATION_RECORD, "lambda=10 authorization") != authorization:
        raise RuntimeError("An existing lambda=10 authorization record differs")
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

print("All Block 07 pre-training gates passed.")
print("The DAN-DG lambda=0.1 prerequisite completed and was reviewed.")
print("Starting DAN-DG controlled-study run with lambda_DG = 10.")
print("Everything except lambda_DG is identical to the lambda=0.1 and lambda=1 runs.")
print("Updates per epoch: 235")
print("Maximum epochs: 30")
print("Early-stopping patience: 5")
print("Selection: unweighted mean source-validation macro-F1")
print("Gradient clipping max-norm: 20")
print("Non-finite values will stop the run; finite weak results will be preserved.")
print("Sketch images accessible to training: 0")
print("No SAM run will start from this block.\n")
subprocess.run(command, cwd=CODE_ROOT, check=True)

# Audit a normally completed lambda=10 run.
run_manifest_path = RUN_DIRECTORY / "run.json"
history_path = RUN_DIRECTORY / "history.csv"
best_validation_path = RUN_DIRECTORY / "best_source_validation.json"
best_checkpoint_path = RUN_DIRECTORY / "best.pt"
run_manifest = load_json(run_manifest_path, "lambda=10 run manifest")
require_fields(
    run_manifest,
    {
        "status": "TASK3_RUN_COMPLETE",
        "run_id": RUN_ID,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "steps_per_source_epoch": 235,
    },
    "lambda=10 run manifest",
)
identity = run_manifest.get("identity")
if not isinstance(identity, dict):
    raise RuntimeError("The lambda=10 run manifest has no identity")
config = identity.get("config")
if not isinstance(config, dict):
    raise RuntimeError("The lambda=10 run identity has no configuration")
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
        "mmd_bandwidth_pairs": "strict_upper_triangle_keep_off_diagonal_zeros",
        "mmd_kernel": "exp(-squared_distance/(2*bandwidth))",
        "mmd_estimator": "literal_empirical_mean_embedding_v_statistic",
    },
    "lambda=10 configuration",
)
require_fields(
    identity,
    {
        "code_sha256": EXPECTED_CODE_TREE_SHA256,
        "code_preflight_sha256": EXPECTED_CODE_PREFLIGHT_SHA256,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
    },
    "lambda=10 run identity",
)

epochs_completed = run_manifest.get("epochs_completed")
best_epoch = run_manifest.get("best_epoch")
best_f1 = run_manifest.get("best_mean_source_macro_f1")
if not isinstance(epochs_completed, int) or not 1 <= epochs_completed <= 30:
    raise RuntimeError("The lambda=10 run recorded an invalid epoch count")
if not isinstance(best_epoch, int) or not 1 <= best_epoch <= epochs_completed:
    raise RuntimeError("The lambda=10 run recorded an invalid best epoch")
if not isinstance(best_f1, (int, float)) or not math.isfinite(float(best_f1)):
    raise RuntimeError("The lambda=10 run recorded an invalid selected macro-F1")

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
    raise FileNotFoundError("The selected lambda=10 checkpoint is missing")
best_checkpoint_sha256 = sha256_file(best_checkpoint_path)
if best_checkpoint_sha256 != run_manifest.get("best_checkpoint_sha256"):
    raise RuntimeError("The selected lambda=10 checkpoint hash differs from run.json")
if not history_path.is_file():
    raise FileNotFoundError("The lambda=10 history is missing")
with history_path.open(newline="") as handle:
    history = list(csv.DictReader(handle))
if len(history) != epochs_completed:
    raise RuntimeError("The lambda=10 history length differs from run.json")
if [int(row["epoch"]) for row in history] != list(range(1, epochs_completed + 1)):
    raise RuntimeError("The lambda=10 history epochs are not consecutive")

completion = {
    "status": "TASK3_BLOCK_07_DAN_DG_10_PASS",
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

print("\nDAN-DG lambda=10 epoch summary:")
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
print("Send the complete final output for review before starting SAM.")
