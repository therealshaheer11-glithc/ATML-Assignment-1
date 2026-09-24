"""Task 3 notebook block 04: install and verify the source-only implementation.

Prerequisites:
  * Blocks 01, 02, and 03 passed in this runtime.
  * The reviewed Task 3 files have been committed to the repository's default branch.

This block fast-forwards the existing Colab checkout, runs the target-free unit tests,
and executes the read-only code/data/artifact preflight. It does not train a model and
does not traverse, extract, or open Sketch.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROTOCOL_VERSION = "task3-approved-2026-09-24-v1"
EXPECTED_TEST_COUNT = 14
EXPECTED_CODE_TREE_SHA256 = (
    "4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44"
)

CODE_ROOT = Path("/content/atml_pa1_task3_source")
SOURCE_ROOT = Path("/content/task3_pacs_sources_v1")

ATML_DRIVE_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_DRIVE_ROOT = ATML_DRIVE_ROOT / "task3_domain_generalization_20260924"
PROVENANCE_ROOT = TASK3_DRIVE_ROOT / "provenance"
SOURCE_PROTOCOL = (
    TASK3_DRIVE_ROOT / "source_protocol" / "pacs_sources_seed6304.json"
)

TASK2_ROOT = ATML_DRIVE_ROOT / "task2_corrected_normalized_v3_20260923"
COMMON_INITIALIZATION = (
    TASK2_ROOT / "initialization" / "resnet18_v1_seed6304_common.pt"
)
ERM_CHECKPOINT = TASK2_ROOT / "source_only" / "best.pt"

BLOCK02_RECORD = PROVENANCE_ROOT / "task2_artifact_verification.json"
BLOCK03_RECORD = PROVENANCE_ROOT / "source_data_preparation.json"
CODE_PREFLIGHT_RECORD = PROVENANCE_ROOT / "code_preflight.json"
IMPLEMENTATION_RECORD = PROVENANCE_ROOT / "implementation_verification.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_passing_record(
    path: Path,
    expected_status: str,
    sketch_count_field: str,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required prior record is missing: {path}")
    record = json.loads(path.read_text())
    if record.get("status") != expected_status:
        raise RuntimeError(f"Prior record did not pass: {path}")
    if record.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Prior record uses a different protocol: {path}")
    if record.get("source_only_phase") is not True:
        raise RuntimeError(f"Prior record does not preserve source-only status: {path}")
    if record.get(sketch_count_field) != 0:
        raise RuntimeError(
            f"Prior record reports Sketch access in {sketch_count_field!r}: {path}"
        )
    return record


load_passing_record(
    BLOCK02_RECORD,
    "TASK3_BLOCK_02_PASS",
    "sketch_images_accessed",
)
load_passing_record(
    BLOCK03_RECORD,
    "TASK3_BLOCK_03_PASS",
    "sketch_images_opened_or_extracted",
)

if not (CODE_ROOT / ".git").is_dir():
    raise FileNotFoundError(f"Block 03 repository checkout is missing: {CODE_ROOT}")

dirty = subprocess.check_output(
    ["git", "status", "--porcelain"],
    cwd=CODE_ROOT,
    text=True,
).strip()
if dirty:
    raise RuntimeError(
        "The Colab repository checkout has uncommitted changes. Preserve and inspect "
        "them instead of overwriting them:\n" + dirty
    )

subprocess.run(["git", "pull", "--ff-only"], cwd=CODE_ROOT, check=True)
repository_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=CODE_ROOT,
    text=True,
).strip()

required_files = (
    CODE_ROOT / "shared" / "mmd.py",
    CODE_ROOT / "task3" / "config.py",
    CODE_ROOT / "task3" / "data.py",
    CODE_ROOT / "task3" / "model.py",
    CODE_ROOT / "task3" / "methods" / "dan_dg.py",
    CODE_ROOT / "task3" / "methods" / "sam.py",
    CODE_ROOT / "task3" / "preflight.py",
    CODE_ROOT / "task3" / "train.py",
    CODE_ROOT / "task3" / "tests" / "test_task3_core.py",
    CODE_ROOT / "task3" / "preregistration" / "DAN_DG_STRENGTH_EXPECTATION.md",
)
missing = [str(path) for path in required_files if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Committed Task 3 implementation is incomplete: {missing}")

compile_result = subprocess.run(
    [sys.executable, "-m", "compileall", "-q", "task3"],
    cwd=CODE_ROOT,
    text=True,
    capture_output=True,
)
if compile_result.returncode != 0:
    print(compile_result.stdout)
    print(compile_result.stderr, file=sys.stderr)
    raise RuntimeError("Task 3 source compilation failed")

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
    text=True,
    capture_output=True,
)
print(test_result.stdout, end="")
print(test_result.stderr, end="", file=sys.stderr)
if test_result.returncode != 0:
    raise RuntimeError("Task 3 target-free unit tests failed")

test_output = test_result.stdout + "\n" + test_result.stderr
match = re.search(r"Ran\s+(\d+)\s+tests?", test_output)
if match is None:
    raise RuntimeError("Could not verify how many Task 3 unit tests ran")
test_count = int(match.group(1))
if test_count != EXPECTED_TEST_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_TEST_COUNT} Task 3 tests, but ran {test_count}"
    )

preflight_command = [
    sys.executable,
    "-m",
    "task3.preflight",
    "--code-root",
    str(CODE_ROOT),
    "--pacs-source-root",
    str(SOURCE_ROOT),
    "--protocol",
    str(SOURCE_PROTOCOL),
    "--initialization",
    str(COMMON_INITIALIZATION),
    "--erm-checkpoint",
    str(ERM_CHECKPOINT),
    "--preregistration",
    str(CODE_ROOT / "task3" / "preregistration" / "DAN_DG_STRENGTH_EXPECTATION.md"),
    "--output",
    str(CODE_PREFLIGHT_RECORD),
]
preflight_result = subprocess.run(
    preflight_command,
    cwd=CODE_ROOT,
    text=True,
    capture_output=True,
)
print(preflight_result.stdout, end="")
print(preflight_result.stderr, end="", file=sys.stderr)
if preflight_result.returncode != 0:
    raise RuntimeError("Task 3 code preflight failed")

preflight = json.loads(CODE_PREFLIGHT_RECORD.read_text())
if preflight.get("status") != "TASK3_CODE_PREFLIGHT_PASS":
    raise RuntimeError("Task 3 code preflight did not record a pass")
if preflight.get("training_started") is not False:
    raise RuntimeError("Code preflight unexpectedly reports that training started")
if preflight.get("source_only_phase") is not True:
    raise RuntimeError("Code preflight did not preserve source-only status")
if preflight.get("sketch_images_accessed") != 0:
    raise RuntimeError("Code preflight reports unexpected Sketch access")
if preflight.get("code_tree_sha256") != EXPECTED_CODE_TREE_SHA256:
    raise RuntimeError(
        "Installed executable code differs from the reviewed Task 3 package: "
        f"{preflight.get('code_tree_sha256')} != {EXPECTED_CODE_TREE_SHA256}"
    )

implementation_record = {
    "status": "TASK3_BLOCK_04_PASS",
    "protocol_version": PROTOCOL_VERSION,
    "source_only_phase": True,
    "training_started": False,
    "sketch_images_accessed": 0,
    "repository": {
        "path": str(CODE_ROOT),
        "commit": repository_commit,
    },
    "unit_tests": {
        "count": test_count,
        "passed": test_count,
        "failed": 0,
    },
    "code_tree_sha256": preflight["code_tree_sha256"],
    "code_preflight": {
        "path": str(CODE_PREFLIGHT_RECORD),
        "sha256": sha256_file(CODE_PREFLIGHT_RECORD),
    },
}

if IMPLEMENTATION_RECORD.exists():
    existing = json.loads(IMPLEMENTATION_RECORD.read_text())
    if existing != implementation_record:
        raise RuntimeError(
            "A different implementation-verification record already exists"
        )
else:
    atomic_write_json(implementation_record, IMPLEMENTATION_RECORD)

print(json.dumps(implementation_record, indent=2))
print(f"\nSaved implementation record: {IMPLEMENTATION_RECORD}")
print("Training started: False")
print("Sketch images accessed: 0")
print("Next action: inspect this record before authorizing the first training run.")
