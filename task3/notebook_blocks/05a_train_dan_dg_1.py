# TASK 3 — BLOCK 05
# Train the prescribed main DAN-DG model: lambda_DG = 1.
#
# This cell:
#   - requires the passing Block 04 records;
#   - verifies the exact repository commit and preflight identity;
#   - refuses to overwrite or silently resume an existing run;
#   - uses only Photo, Art Painting, and Cartoon;
#   - never loads Sketch.

from pathlib import Path
import hashlib
import json
import subprocess
import sys


RUN_ID = "dan_dg_1"

EXPECTED_COMMIT = "19208b4c62acb980fb3246f30e062784b90d8dfc"
EXPECTED_CODE_TREE_SHA256 = (
    "4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44"
)
EXPECTED_CODE_PREFLIGHT_SHA256 = (
    "40ef37d7ae08ece5526e5588e446e5301158cc4bd444e2213cedfc0a9bf73eee"
)

CODE_ROOT = Path("/content/atml_pa1_task3_source")
SOURCE_ROOT = Path("/content/task3_pacs_sources_v1")

ATML_DRIVE_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_ROOT = (
    ATML_DRIVE_ROOT
    / "task3_domain_generalization_20260924"
)
PROVENANCE_ROOT = TASK3_ROOT / "provenance"
SOURCE_PROTOCOL = (
    TASK3_ROOT
    / "source_protocol"
    / "pacs_sources_seed6304.json"
)
TRAINING_ROOT = TASK3_ROOT / "training"
RUN_DIRECTORY = TRAINING_ROOT / RUN_ID

TASK2_ROOT = (
    ATML_DRIVE_ROOT
    / "task2_corrected_normalized_v3_20260923"
)
COMMON_INITIALIZATION = (
    TASK2_ROOT
    / "initialization"
    / "resnet18_v1_seed6304_common.pt"
)

PREREGISTRATION = (
    CODE_ROOT
    / "task3"
    / "preregistration"
    / "DAN_DG_STRENGTH_EXPECTATION.md"
)
CODE_PREFLIGHT = PROVENANCE_ROOT / "code_preflight.json"
IMPLEMENTATION_RECORD = (
    PROVENANCE_ROOT
    / "implementation_verification.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


# Gate 1: require the passing Block 04 record.
if not IMPLEMENTATION_RECORD.is_file():
    raise FileNotFoundError(
        f"Missing Block 04 record: {IMPLEMENTATION_RECORD}"
    )

implementation = json.loads(
    IMPLEMENTATION_RECORD.read_text()
)

required_fields = {
    "status": "TASK3_BLOCK_04_PASS",
    "protocol_version": "task3-approved-2026-09-24-v1",
    "source_only_phase": True,
    "training_started": False,
    "sketch_images_accessed": 0,
    "code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
}

for field, expected in required_fields.items():
    actual = implementation.get(field)
    if actual != expected:
        raise RuntimeError(
            f"Block 04 field {field!r} differs: "
            f"{actual!r} != {expected!r}"
        )

if implementation.get("unit_tests") != {
    "count": 14,
    "passed": 14,
    "failed": 0,
}:
    raise RuntimeError(
        "Block 04 did not record all 14 tests passing"
    )


# Gate 2: verify the persistent code-preflight file.
if not CODE_PREFLIGHT.is_file():
    raise FileNotFoundError(
        f"Missing code preflight: {CODE_PREFLIGHT}"
    )

actual_preflight_sha256 = sha256_file(CODE_PREFLIGHT)

if actual_preflight_sha256 != EXPECTED_CODE_PREFLIGHT_SHA256:
    raise RuntimeError(
        "Code-preflight identity changed:\n"
        f"Expected: {EXPECTED_CODE_PREFLIGHT_SHA256}\n"
        f"Actual:   {actual_preflight_sha256}"
    )

if (
    implementation["code_preflight"]["sha256"]
    != actual_preflight_sha256
):
    raise RuntimeError(
        "Implementation record and code preflight disagree"
    )


# Gate 3: prevent code changes after Block 04.
current_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=CODE_ROOT,
    text=True,
).strip()

if current_commit != EXPECTED_COMMIT:
    raise RuntimeError(
        "Repository commit changed after Block 04:\n"
        f"Expected: {EXPECTED_COMMIT}\n"
        f"Actual:   {current_commit}"
    )

if (
    implementation["repository"]["commit"]
    != current_commit
):
    raise RuntimeError(
        "Implementation record uses a different commit"
    )


# Gate 4: never overwrite or silently resume a run.
if RUN_DIRECTORY.exists():
    if (RUN_DIRECTORY / "run.json").is_file():
        raise FileExistsError(
            "DAN-DG lambda=1 is already complete. "
            "Do not retrain it."
        )

    if (RUN_DIRECTORY / "last.pt").is_file():
        raise RuntimeError(
            "A completed-epoch checkpoint already exists. "
            "Do not overwrite or silently resume it. "
            "Send me the current state so we can use the "
            "controlled resume procedure."
        )

    raise RuntimeError(
        "An unexpected incomplete DAN-DG run directory "
        f"already exists: {RUN_DIRECTORY}"
    )


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

print("All Block 05 pre-training gates passed.")
print("Starting DAN-DG with lambda_DG = 1.")
print("Updates per epoch: 235")
print("Maximum epochs: 30")
print("Early-stopping patience: 5")
print("Selection: mean source-validation macro-F1")
print("Sketch images accessible to training: 0")
print(
    "The first epoch record will appear after source "
    "verification and one complete training epoch.\n"
)

completed = subprocess.run(
    command,
    cwd=CODE_ROOT,
    check=True,
)

print(
    "\nDAN-DG lambda=1 training process completed "
    f"with return code {completed.returncode}."
)
print(
    "Send me the complete final output before starting "
    "another run."
)
