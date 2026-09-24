"""Task 3 notebook block 08: train the prescribed SAM main comparison.

Paste this file's complete contents into one Colab cell. It authenticates the frozen
implementation and all three completed DAN-DG conditions, launches only standard
non-adaptive SAM at rho=0.05, audits successful outputs, and stops for source-only
review. It never loads Sketch and does not begin final evaluation.
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

EXPECTED_DAN_RUNS = {
    "dan_dg_0p1": {
        "record": "dan_dg_0p1_training_completion.json",
        "status": "TASK3_BLOCK_06_DAN_DG_0P1_PASS",
        "lambda": 0.1,
        "epochs": 10,
        "best_epoch": 5,
        "best_f1": 0.9462297763360951,
        "checkpoint_sha256": (
            "dc6036a28e3af8c281b143adb6f47b4d03676fa44c117e912ef9bcd959d8ca27"
        ),
        "history_sha256": (
            "fc92efd81ea529a48373977f9d4973e754eb53cfca7622c8513ad5cfc64b35b8"
        ),
        "manifest_sha256": (
            "731f07bded36f6034e014eb217dca2b1b3a9c7836ed0e5bd72d0c2be3537672a"
        ),
    },
    "dan_dg_1": {
        "record": "dan_dg_1_training_completion.json",
        "status": "TASK3_BLOCK_05_DAN_DG_1_PASS",
        "lambda": 1.0,
        "epochs": 8,
        "best_epoch": 3,
        "best_f1": 0.8693286334613551,
        "checkpoint_sha256": (
            "a44bff8e519134459c2d6bf056785801f9944945d160e516b4da7f5e2752992b"
        ),
        "history_sha256": (
            "321e267d9b628e357811da617fc5dac07149bdcbd30db030f4c8b65c20518ed7"
        ),
        "manifest_sha256": (
            "33eb66823509417c758095c820efebdce2b8d90b64a3eecde64e3eaecea83e2f"
        ),
    },
    "dan_dg_10": {
        "record": "dan_dg_10_training_completion.json",
        "status": "TASK3_BLOCK_07_DAN_DG_10_PASS",
        "lambda": 10.0,
        "epochs": 12,
        "best_epoch": 7,
        "best_f1": 0.05066996495567924,
        "checkpoint_sha256": (
            "f8cc723dc16b5e17a48f8454541beb473b70e6c38d45895f170c09f59821b2c8"
        ),
        "history_sha256": (
            "dec8773f2a0642b2aecea85619dbd8bdc19a74723c06a5ce9ca24ce9c8b4b918"
        ),
        "manifest_sha256": (
            "962626d182952f068a689bb213474074f8f07869c9a5a254cc246164d8fe4330"
        ),
    },
}

RUN_ID = "sam"
EXPECTED_RHO = 0.05

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
AUTHORIZATION_RECORD = PROVENANCE_ROOT / "sam_training_authorization.json"
COMPLETION_RECORD = PROVENANCE_ROOT / "sam_training_completion.json"


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


def verify_artifact(entry: dict, expected_sha256: str, description: str) -> None:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise RuntimeError(f"Missing recorded {description} path")
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Recorded {description} is missing: {path}")
    actual = sha256_file(path)
    if entry.get("sha256") != expected_sha256 or actual != expected_sha256:
        raise RuntimeError(f"Recorded {description} hash differs")


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

# Gate 2: authenticate all completed DAN-DG training conditions.
dan_record_identities = {}
for run_id, expected in EXPECTED_DAN_RUNS.items():
    record_path = PROVENANCE_ROOT / expected["record"]
    record = load_json(record_path, f"{run_id} completion record")
    require_fields(
        record,
        {
            "status": expected["status"],
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "method": "dan_dg",
            "mmd_lambda": expected["lambda"],
            "source_only_phase": True,
            "training_completed": True,
            "sketch_images_accessed": 0,
            "repository_commit": EXPECTED_REPOSITORY_COMMIT,
            "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
            "epochs_completed": expected["epochs"],
            "best_epoch": expected["best_epoch"],
            "best_mean_source_macro_f1": expected["best_f1"],
        },
        f"{run_id} completion record",
    )
    verify_artifact(
        record.get("best_checkpoint"),
        expected["checkpoint_sha256"],
        f"{run_id} selected checkpoint",
    )
    verify_artifact(
        record.get("history"), expected["history_sha256"], f"{run_id} history"
    )
    verify_artifact(
        record.get("run_manifest"),
        expected["manifest_sha256"],
        f"{run_id} run manifest",
    )
    dan_record_identities[run_id] = {
        "path": str(record_path),
        "sha256": sha256_file(record_path),
    }

# Gate 3: refuse overwrite, implicit resume, or an already completed SAM run.
if COMPLETION_RECORD.exists():
    raise FileExistsError(f"The SAM completion record already exists: {COMPLETION_RECORD}")
if RUN_DIRECTORY.exists():
    if (RUN_DIRECTORY / "run.json").is_file():
        raise FileExistsError("The SAM main run is already complete; do not rerun it")
    if (RUN_DIRECTORY / "last.pt").is_file():
        raise RuntimeError(
            "A completed-epoch SAM checkpoint already exists. Do not silently resume "
            "it; preserve the directory and request a reviewed resume block."
        )
    raise RuntimeError(f"Unexpected SAM run directory: {RUN_DIRECTORY}")

# Persist the approved fixed SAM design before launching it.
authorization = {
    "status": "TASK3_SAM_MAIN_AUTHORIZED",
    "protocol_version": PROTOCOL_VERSION,
    "student_approval_received": True,
    "dan_dg_strength_study_completed_and_reviewed": True,
    "approved_next_run": RUN_ID,
    "method": "standard_non_adaptive_sam",
    "sam_rho": EXPECTED_RHO,
    "objective": "source_erm_classification_only",
    "first_pass_gradient_clipping": False,
    "second_pass_update_gradient_clipping_max_norm": 20.0,
    "batchnorm_running_statistics_frozen_during_both_passes": True,
    "all_shared_settings_unchanged": True,
    "source_only_phase": True,
    "sketch_images_accessed": 0,
    "repository_commit": current_commit,
    "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
    "completed_dan_dg_records": dan_record_identities,
}
if AUTHORIZATION_RECORD.exists():
    if load_json(AUTHORIZATION_RECORD, "SAM authorization") != authorization:
        raise RuntimeError("An existing SAM authorization record differs")
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

print("All Block 08 pre-training gates passed.")
print("All three DAN-DG conditions are complete and authenticated.")
print("Starting prescribed standard non-adaptive SAM main run with rho = 0.05.")
print("Objective: source ERM classification loss only")
print("Updates per epoch: 235, with two forward/backward passes per update")
print("Maximum epochs: 30")
print("Early-stopping patience: 5")
print("Selection: unweighted mean source-validation macro-F1")
print("First-pass gradient clipping: disabled")
print("Second-pass update-gradient clipping max-norm: 20")
print("BatchNorm running statistics frozen during both passes: yes")
print("Sketch images accessible to training: 0")
print("No final evaluation will start from this block.\n")
subprocess.run(command, cwd=CODE_ROOT, check=True)

# Audit a normally completed SAM run.
run_manifest_path = RUN_DIRECTORY / "run.json"
history_path = RUN_DIRECTORY / "history.csv"
best_validation_path = RUN_DIRECTORY / "best_source_validation.json"
best_checkpoint_path = RUN_DIRECTORY / "best.pt"
run_manifest = load_json(run_manifest_path, "SAM run manifest")
require_fields(
    run_manifest,
    {
        "status": "TASK3_RUN_COMPLETE",
        "run_id": RUN_ID,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "steps_per_source_epoch": 235,
    },
    "SAM run manifest",
)
identity = run_manifest.get("identity")
if not isinstance(identity, dict):
    raise RuntimeError("The SAM run manifest has no identity")
config = identity.get("config")
if not isinstance(config, dict):
    raise RuntimeError("The SAM run identity has no configuration")
require_fields(
    config,
    {
        "run_id": RUN_ID,
        "method": "sam",
        "mmd_lambda": None,
        "sam_rho": EXPECTED_RHO,
        "sam_variant": "standard_non_adaptive",
        "sam_first_pass_clipping": False,
        "sam_second_pass_clipping": True,
        "gradient_clipping": 20.0,
        "source_batch_per_domain": 8,
        "steps_per_source_epoch": 235,
        "epochs": 30,
        "patience": 5,
        "selection_metric": "unweighted_mean_source_validation_macro_f1",
        "batchnorm_running_statistics": "frozen_at_imagenet_values",
        "batchnorm_affine_parameters": "trainable",
    },
    "SAM configuration",
)
require_fields(
    identity,
    {
        "code_sha256": EXPECTED_CODE_TREE_SHA256,
        "code_preflight_sha256": EXPECTED_CODE_PREFLIGHT_SHA256,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
    },
    "SAM run identity",
)

epochs_completed = run_manifest.get("epochs_completed")
best_epoch = run_manifest.get("best_epoch")
best_f1 = run_manifest.get("best_mean_source_macro_f1")
if not isinstance(epochs_completed, int) or not 1 <= epochs_completed <= 30:
    raise RuntimeError("The SAM run recorded an invalid epoch count")
if not isinstance(best_epoch, int) or not 1 <= best_epoch <= epochs_completed:
    raise RuntimeError("The SAM run recorded an invalid best epoch")
if not isinstance(best_f1, (int, float)) or not math.isfinite(float(best_f1)):
    raise RuntimeError("The SAM run recorded an invalid selected macro-F1")

best_validation = load_json(best_validation_path, "selected SAM source validation")
if best_validation.get("epoch") != best_epoch:
    raise RuntimeError("Selected SAM source-validation epoch differs from run.json")
if not math.isclose(
    float(best_validation.get("mean_source_macro_f1")),
    float(best_f1),
    rel_tol=0.0,
    abs_tol=1e-15,
):
    raise RuntimeError("Selected SAM source macro-F1 differs from run.json")

if not best_checkpoint_path.is_file():
    raise FileNotFoundError("The selected SAM checkpoint is missing")
best_checkpoint_sha256 = sha256_file(best_checkpoint_path)
if best_checkpoint_sha256 != run_manifest.get("best_checkpoint_sha256"):
    raise RuntimeError("The selected SAM checkpoint hash differs from run.json")
if not history_path.is_file():
    raise FileNotFoundError("The SAM history is missing")
with history_path.open(newline="") as handle:
    history = list(csv.DictReader(handle))
if len(history) != epochs_completed:
    raise RuntimeError("The SAM history length differs from run.json")
if [int(row["epoch"]) for row in history] != list(range(1, epochs_completed + 1)):
    raise RuntimeError("The SAM history epochs are not consecutive")

completion = {
    "status": "TASK3_BLOCK_08_SAM_PASS",
    "protocol_version": PROTOCOL_VERSION,
    "run_id": RUN_ID,
    "method": "sam",
    "sam_variant": "standard_non_adaptive",
    "sam_rho": EXPECTED_RHO,
    "source_only_phase": True,
    "training_completed": True,
    "review_required_before_diagnostics": True,
    "final_evaluation_started": False,
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

print("\nSAM epoch summary:")
for row in history:
    print(
        f"epoch={int(row['epoch']):02d} "
        f"base_classification={float(row['classification_loss']):.6f} "
        f"perturbed_classification="
        f"{float(row['sam_perturbed_classification_loss']):.6f} "
        f"first_gradient={float(row['sam_first_gradient_norm']):.6f} "
        f"perturbation={float(row['sam_perturbation_norm']):.6f} "
        f"update_gradient={float(row['gradient_norm']):.6f} "
        f"clip_fraction={float(row['gradient_clipped_fraction']):.4f} "
        f"mean_f1={float(row['mean_source_macro_f1']):.6f} "
        f"worst_f1={float(row['worst_source_macro_f1']):.6f}"
    )

print("\nAudited completion record:\n")
print(json.dumps(completion, indent=2))
print(f"\nSaved at: {COMPLETION_RECORD}")
print("Sketch images accessed: 0")
print("Final evaluation started: False")
print("Send the complete final output for source-only review before diagnostics.")
