"""Task 3 notebook block 09: install and run locked source-only diagnostics.

Run only after the listed post-training files have been committed through GitHub web.
This block fast-forwards the clean Colab checkout, verifies the exact allowed diff and
diagnostic code tree, runs all 18 target-free tests, authenticates the completed SAM
run, executes the source-domain probe and common sharpness proxy, and persists an audit.
It never loads Sketch and does not create the final experiment lock.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PROTOCOL_VERSION = "task3-approved-2026-09-24-v1"
TRAINING_COMMIT = "19208b4c62acb980fb3246f30e062784b90d8dfc"
TRAINING_CODE_SHA256 = (
    "4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44"
)
EXPECTED_DIAGNOSTIC_CODE_SHA256 = (
    "df3de3b753a6008ed8ccae50d4ef8f278165dcc1bbdfe57e9acb005ce08f1989"
)
EXPECTED_SAM_CHECKPOINT_SHA256 = (
    "040a80ce15974d71f3a90b5aab5050965d27212822079e5a075c742052c0827d"
)
EXPECTED_SAM_HISTORY_SHA256 = (
    "d3150a25d014de91c8f349a35dc17ecbac30381920360e0a0438977bf2d1448e"
)
EXPECTED_SAM_MANIFEST_SHA256 = (
    "d2a1a76200c6af0aed54c5f079d4619be2ab1ddf11c8c723c7f7f2ad12e9dd5e"
)

REQUIRED_POST_TRAINING_UPDATE = {
    "task3/README.md",
    "task3/docs/TASK3_PROTOCOL_AND_DECISIONS.md",
    "task3/provenance/RUN_LOG.md",
    "task3/evaluation/__init__.py",
    "task3/evaluation/run_source_diagnostics.py",
    "task3/evaluation/sharpness.py",
    "task3/evaluation/source_domain_separability.py",
    "task3/tests/test_task3_diagnostics.py",
    "task3/notebook_blocks/05a_train_dan_dg_1.py",
    "task3/notebook_blocks/05b_audit_dan_dg_1.py",
    "task3/notebook_blocks/05c_diagnose_dan_dg_1_instability.py",
    "task3/notebook_blocks/06_train_dan_dg_0p1.py",
    "task3/notebook_blocks/07_train_dan_dg_10.py",
    "task3/notebook_blocks/08_train_sam.py",
    "task3/notebook_blocks/09_install_and_run_source_diagnostics.py",
}

CODE_ROOT = Path("/content/atml_pa1_task3_source")
SOURCE_ROOT = Path("/content/task3_pacs_sources_v1")
ATML_DRIVE_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_ROOT = ATML_DRIVE_ROOT / "task3_domain_generalization_20260924"
PROVENANCE_ROOT = TASK3_ROOT / "provenance"
SOURCE_PROTOCOL = TASK3_ROOT / "source_protocol" / "pacs_sources_seed6304.json"
TRAINING_ROOT = TASK3_ROOT / "training"
TASK2_ROOT = ATML_DRIVE_ROOT / "task2_corrected_normalized_v3_20260923"
ERM_CHECKPOINT = TASK2_ROOT / "source_only" / "best.pt"
SAM_COMPLETION = PROVENANCE_ROOT / "sam_training_completion.json"
AUTHORIZATION_RECORD = PROVENANCE_ROOT / "source_diagnostics_authorization.json"
DIAGNOSTICS_ROOT = TASK3_ROOT / "source_diagnostics"
DIAGNOSTICS_RESULT = DIAGNOSTICS_ROOT / "source_diagnostics.json"
COMPLETION_RECORD = PROVENANCE_ROOT / "source_diagnostics_completion.json"


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
        raise RuntimeError(f"{description} is not a JSON object")
    return payload


def verify_artifact(entry: dict, expected_sha256: str, description: str) -> None:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise RuntimeError(f"SAM completion lacks {description}")
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"SAM {description} is missing: {path}")
    if entry.get("sha256") != expected_sha256 or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"SAM {description} hash differs")


# Gate 1: update only a clean checkout and prove it descends from the training commit.
if not (CODE_ROOT / ".git").is_dir():
    raise FileNotFoundError(f"Colab repository checkout is missing: {CODE_ROOT}")
dirty = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True
).strip()
if dirty:
    raise RuntimeError("The Colab checkout has uncommitted changes:\n" + dirty)
subprocess.run(["git", "pull", "--ff-only"], cwd=CODE_ROOT, check=True)
diagnostic_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
).strip()
ancestor_check = subprocess.run(
    ["git", "merge-base", "--is-ancestor", TRAINING_COMMIT, diagnostic_commit],
    cwd=CODE_ROOT,
)
if ancestor_check.returncode != 0 or diagnostic_commit == TRAINING_COMMIT:
    raise RuntimeError("Diagnostic commit is not a strict descendant of training commit")
changed_files = set(
    filter(
        None,
        subprocess.check_output(
            ["git", "diff", "--name-only", f"{TRAINING_COMMIT}..{diagnostic_commit}"],
            cwd=CODE_ROOT,
            text=True,
        ).splitlines(),
    )
)
missing = REQUIRED_POST_TRAINING_UPDATE - changed_files
unexpected = changed_files - REQUIRED_POST_TRAINING_UPDATE
if missing or unexpected:
    raise RuntimeError(
        "Post-training Git update differs from the approved file set. "
        f"Missing={sorted(missing)}, unexpected={sorted(unexpected)}"
    )

# Gate 2: exact diagnostic implementation and complete target-free test suite.
sys.path.insert(0, str(CODE_ROOT))
from task3.config import code_tree_sha256  # noqa: E402

actual_diagnostic_code_sha256 = code_tree_sha256(CODE_ROOT)
if actual_diagnostic_code_sha256 != EXPECTED_DIAGNOSTIC_CODE_SHA256:
    raise RuntimeError(
        "Diagnostic code tree differs. "
        f"{actual_diagnostic_code_sha256} != {EXPECTED_DIAGNOSTIC_CODE_SHA256}"
    )
test_result = subprocess.run(
    [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "task3/tests",
        "-v",
    ],
    cwd=CODE_ROOT,
)
if test_result.returncode != 0:
    raise RuntimeError("The post-training target-free test suite failed")

# Gate 3: authenticate the selected SAM artifact and preserve source-only status.
sam = load_json(SAM_COMPLETION, "SAM completion record")
for key, expected in {
    "status": "TASK3_BLOCK_08_SAM_PASS",
    "protocol_version": PROTOCOL_VERSION,
    "run_id": "sam",
    "method": "sam",
    "sam_variant": "standard_non_adaptive",
    "sam_rho": 0.05,
    "source_only_phase": True,
    "training_completed": True,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "repository_commit": TRAINING_COMMIT,
    "code_tree_sha256": TRAINING_CODE_SHA256,
    "epochs_completed": 10,
    "best_epoch": 5,
    "best_mean_source_macro_f1": 0.9540473940355131,
}.items():
    if sam.get(key) != expected:
        raise RuntimeError(f"SAM completion field differs: {key}")
verify_artifact(sam.get("best_checkpoint"), EXPECTED_SAM_CHECKPOINT_SHA256, "checkpoint")
verify_artifact(sam.get("history"), EXPECTED_SAM_HISTORY_SHA256, "history")
verify_artifact(sam.get("run_manifest"), EXPECTED_SAM_MANIFEST_SHA256, "manifest")

if COMPLETION_RECORD.exists() or DIAGNOSTICS_RESULT.exists():
    raise FileExistsError("Source diagnostics are already complete; do not rerun them")

authorization = {
    "status": "TASK3_SOURCE_DIAGNOSTICS_AUTHORIZED",
    "protocol_version": PROTOCOL_VERSION,
    "all_required_training_complete": True,
    "checkpoints_immutable": True,
    "approved_diagnostics": [
        "source_domain_separability",
        "common_local_sharpness_proxy",
    ],
    "source_only_phase": True,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "training_commit": TRAINING_COMMIT,
    "training_code_tree_sha256": TRAINING_CODE_SHA256,
    "diagnostic_commit": diagnostic_commit,
    "diagnostic_code_tree_sha256": actual_diagnostic_code_sha256,
    "sam_completion": {
        "path": str(SAM_COMPLETION),
        "sha256": sha256_file(SAM_COMPLETION),
    },
}
if AUTHORIZATION_RECORD.exists():
    if load_json(AUTHORIZATION_RECORD, "source-diagnostic authorization") != authorization:
        raise RuntimeError("Existing source-diagnostic authorization differs")
else:
    atomic_write_json(authorization, AUTHORIZATION_RECORD)

command = [
    sys.executable,
    "-m",
    "task3.evaluation.run_source_diagnostics",
    "--code-root",
    str(CODE_ROOT),
    "--pacs-source-root",
    str(SOURCE_ROOT),
    "--protocol",
    str(SOURCE_PROTOCOL),
    "--erm-checkpoint",
    str(ERM_CHECKPOINT),
    "--training-root",
    str(TRAINING_ROOT),
    "--output",
    str(DIAGNOSTICS_RESULT),
]

print("All Block 09 pre-diagnostic gates passed.")
print(f"Training commit preserved: {TRAINING_COMMIT}")
print(f"Diagnostic commit: {diagnostic_commit}")
print("Target-free tests passed: 18/18")
print("Probe: 334 validation examples per source, 1,002 total")
print("Probe features: raw unnormalized 512-D pre-classifier features")
print("Probe split: shared domain-stratified 70/30 split, seed 6304")
print("Sharpness batch: 32 validation examples per source, 96 total")
print("Sharpness radius: 0.05")
print("Models: ERM, prescribed DAN-DG lambda=1, SAM")
print("Sketch images accessible to diagnostics: 0\n")
subprocess.run(command, cwd=CODE_ROOT, check=True)

result = load_json(DIAGNOSTICS_RESULT, "source-diagnostic result")
for key, expected in {
    "status": "TASK3_SOURCE_DIAGNOSTICS_PASS",
    "protocol_version": PROTOCOL_VERSION,
    "source_only_phase": True,
    "diagnostics_completed": True,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "training_code_tree_sha256": TRAINING_CODE_SHA256,
    "diagnostic_code_tree_sha256": EXPECTED_DIAGNOSTIC_CODE_SHA256,
}.items():
    if result.get(key) != expected:
        raise RuntimeError(f"Source-diagnostic result field differs: {key}")
if set(result.get("models", {})) != {"erm", "dan_dg_1", "sam"}:
    raise RuntimeError("Source diagnostics did not report the three main models")

summary = {}
for model_name in ("erm", "dan_dg_1", "sam"):
    model = result["models"][model_name]
    summary[model_name] = {
        "mean_source_macro_f1": model["source_validation_reproduced"][
            "mean_source_macro_f1"
        ],
        "source_domain_separability": model["source_domain_separability"][
            "accuracy"
        ],
        "delta_sharp": model["common_sharpness_proxy"]["delta_sharp"],
    }

completion = {
    "status": "TASK3_BLOCK_09_SOURCE_DIAGNOSTICS_PASS",
    "protocol_version": PROTOCOL_VERSION,
    "source_only_phase": True,
    "diagnostics_completed": True,
    "experiment_lock_created": False,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "training_commit": TRAINING_COMMIT,
    "training_code_tree_sha256": TRAINING_CODE_SHA256,
    "diagnostic_commit": diagnostic_commit,
    "diagnostic_code_tree_sha256": actual_diagnostic_code_sha256,
    "target_free_tests": {"count": 18, "passed": 18, "failed": 0},
    "authorization": {
        "path": str(AUTHORIZATION_RECORD),
        "sha256": sha256_file(AUTHORIZATION_RECORD),
    },
    "result": {
        "path": str(DIAGNOSTICS_RESULT),
        "sha256": sha256_file(DIAGNOSTICS_RESULT),
    },
    "probe_design": result["probe_design"],
    "sharpness_batch_design": result["sharpness_batch_design"],
    "summary": summary,
}
atomic_write_json(completion, COMPLETION_RECORD)

print("\nAudited source-only diagnostic summary:\n")
print(json.dumps(completion, indent=2))
print(f"\nSaved at: {COMPLETION_RECORD}")
print("Sketch images accessed: 0")
print("Experiment lock created: False")
print("Final evaluation started: False")
print("Send the complete output for review before creating the experiment lock.")
