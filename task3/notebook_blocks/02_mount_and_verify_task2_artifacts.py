"""Task 3 notebook block 02: persist preflight and verify Task 2 artifacts.

This block mounts Google Drive and reads only:
  * the Block 01 JSON;
  * the exact Task 2 common-initialization file; and
  * the exact Task 2 Source-only ERM checkpoint.

It does not inspect, extract, traverse, or load PACS/Sketch data.
"""

from google.colab import drive

drive.mount("/content/drive")

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


PROTOCOL_VERSION = "task3-approved-2026-09-24-v1"

EXPECTED_RUNTIME = {
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "numpy": "2.1.3",
    "sklearn": "1.6.1",
    "cuda": "12.8",
    "cudnn": 91900,
    "device_type": "cuda",
    "gpu": "Tesla T4",
}

EXPECTED_PROTOCOL_SHA256 = (
    "e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74"
)
EXPECTED_INITIALIZATION_STATE_SHA256 = (
    "4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3"
)
EXPECTED_ERM_CHECKPOINT_FILE_SHA256 = (
    "3d28a223e4b97b323cb3a20dcb5b7577af96631f2e6ef1f2bc99d53d85761327"
)

EXPECTED_ERM_SOURCE_VALIDATION = {
    "epoch": 4,
    "photo_accuracy": 0.9730538922155688,
    "photo_macro_f1": 0.9681339341150288,
    "art_painting_accuracy": 0.9097560975609756,
    "art_painting_macro_f1": 0.9119171075935996,
    "cartoon_accuracy": 0.9402985074626866,
    "cartoon_macro_f1": 0.947827661029101,
    "mean_source_accuracy": 0.9410361657464104,
    "mean_source_macro_f1": 0.9426262342459099,
}

LOCAL_PREFLIGHT = Path("/content/task3_runtime_preflight.json")
ATML_DRIVE_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_DRIVE_ROOT = ATML_DRIVE_ROOT / "task3_domain_generalization_20260924"
TASK3_PROVENANCE = TASK3_DRIVE_ROOT / "provenance"

TASK2_V3_ROOT = ATML_DRIVE_ROOT / "task2_corrected_normalized_v3_20260923"
COMMON_INITIALIZATION = (
    TASK2_V3_ROOT / "initialization" / "resnet18_v1_seed6304_common.pt"
)
ERM_CHECKPOINT = TASK2_V3_ROOT / "source_only" / "best.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


if not LOCAL_PREFLIGHT.is_file():
    raise FileNotFoundError(
        "Block 01 JSON is missing. Rerun Block 01 in the same Colab runtime."
    )

preflight = json.loads(LOCAL_PREFLIGHT.read_text())

if preflight.get("status") != "TASK3_BLOCK_01_PASS":
    raise RuntimeError("Block 01 did not record a passing status.")
if preflight.get("protocol_version") != PROTOCOL_VERSION:
    raise RuntimeError("Block 01 protocol version differs from the approved version.")
if preflight.get("source_only_phase") is not True:
    raise RuntimeError("The source-only phase is not active.")
if preflight.get("sketch_images_accessed") != 0:
    raise RuntimeError("Block 01 reports unexpected Sketch access.")

actual_runtime = preflight.get("runtime", {})
if actual_runtime != EXPECTED_RUNTIME:
    raise RuntimeError(
        "Task 3 runtime differs from the verified Task 2 environment.\n"
        f"Expected: {json.dumps(EXPECTED_RUNTIME, sort_keys=True)}\n"
        f"Actual:   {json.dumps(actual_runtime, sort_keys=True)}"
    )

if not COMMON_INITIALIZATION.is_file():
    raise FileNotFoundError(
        "The exact Task 2 common initialization is missing:\n"
        f"{COMMON_INITIALIZATION}"
    )
if not ERM_CHECKPOINT.is_file():
    raise FileNotFoundError(
        "The exact selected Task 2 ERM checkpoint is missing:\n"
        f"{ERM_CHECKPOINT}"
    )

# Load only the two trusted, student-created Task 2 artifacts. No dataset is touched.
initialization_payload = torch.load(
    COMMON_INITIALIZATION,
    map_location="cpu",
    weights_only=False,
)

if "state_dict" not in initialization_payload:
    raise RuntimeError("Common-initialization artifact has no state_dict.")

initialization_state_sha256 = state_dict_sha256(
    initialization_payload["state_dict"]
)

if initialization_state_sha256 != EXPECTED_INITIALIZATION_STATE_SHA256:
    raise RuntimeError(
        "Common-initialization state hash mismatch.\n"
        f"Expected: {EXPECTED_INITIALIZATION_STATE_SHA256}\n"
        f"Actual:   {initialization_state_sha256}"
    )

recorded_initialization_hash = initialization_payload.get("state_dict_sha256")
if recorded_initialization_hash != EXPECTED_INITIALIZATION_STATE_SHA256:
    raise RuntimeError(
        "Common-initialization internal identity does not match the approved state."
    )

erm_checkpoint_file_sha256 = sha256_file(ERM_CHECKPOINT)
if erm_checkpoint_file_sha256 != EXPECTED_ERM_CHECKPOINT_FILE_SHA256:
    raise RuntimeError(
        "ERM checkpoint file hash mismatch.\n"
        f"Expected: {EXPECTED_ERM_CHECKPOINT_FILE_SHA256}\n"
        f"Actual:   {erm_checkpoint_file_sha256}"
    )

erm_payload = torch.load(
    ERM_CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

if erm_payload.get("target_labels_used") is not False:
    raise RuntimeError("ERM checkpoint does not certify target-label exclusion.")
if erm_payload.get("epoch") != EXPECTED_ERM_SOURCE_VALIDATION["epoch"]:
    raise RuntimeError("ERM selected epoch differs from the approved epoch.")

erm_identity = erm_payload.get("identity", {})
if erm_identity.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError("ERM checkpoint uses a different source-split protocol.")
if (
    erm_identity.get("initialization_sha256")
    != EXPECTED_INITIALIZATION_STATE_SHA256
):
    raise RuntimeError("ERM checkpoint uses a different common initialization.")

erm_config = erm_identity.get("config", {})
if erm_config.get("run_id") != "source_only":
    raise RuntimeError("The supplied ERM checkpoint is not the Source-only run.")
if erm_config.get("method") != "source_only":
    raise RuntimeError("The supplied ERM checkpoint has the wrong method identity.")

actual_source_validation = {
    "epoch": erm_payload["epoch"],
    **erm_payload.get("source_validation", {}),
}

for name, expected in EXPECTED_ERM_SOURCE_VALIDATION.items():
    actual = actual_source_validation.get(name)
    if actual is None or abs(float(actual) - float(expected)) > 1e-12:
        raise RuntimeError(
            f"ERM source-validation mismatch for {name}: "
            f"expected {expected}, got {actual}"
        )

TASK3_PROVENANCE.mkdir(parents=True, exist_ok=True)

persistent_preflight = TASK3_PROVENANCE / "runtime_preflight.json"
if persistent_preflight.exists():
    if persistent_preflight.read_bytes() != LOCAL_PREFLIGHT.read_bytes():
        raise RuntimeError(
            "A different runtime_preflight.json already exists in Task 3 provenance."
        )
else:
    shutil.copy2(LOCAL_PREFLIGHT, persistent_preflight)

artifact_record = {
    "status": "TASK3_BLOCK_02_PASS",
    "protocol_version": PROTOCOL_VERSION,
    "source_only_phase": True,
    "dataset_traversed_by_this_block": False,
    "sketch_images_accessed": 0,
    "task3_drive_root": str(TASK3_DRIVE_ROOT),
    "runtime_preflight": {
        "path": str(persistent_preflight),
        "sha256": sha256_file(persistent_preflight),
    },
    "common_initialization": {
        "path": str(COMMON_INITIALIZATION),
        "state_dict_sha256": initialization_state_sha256,
        "verified": True,
    },
    "erm_checkpoint": {
        "path": str(ERM_CHECKPOINT),
        "file_sha256": erm_checkpoint_file_sha256,
        "selected_epoch": erm_payload["epoch"],
        "source_validation": erm_payload["source_validation"],
        "protocol_sha256": erm_identity["protocol_sha256"],
        "initialization_state_sha256": erm_identity["initialization_sha256"],
        "target_labels_used": erm_payload["target_labels_used"],
        "verified": True,
    },
}

artifact_record_path = TASK3_PROVENANCE / "task2_artifact_verification.json"
if artifact_record_path.exists():
    existing = json.loads(artifact_record_path.read_text())
    if existing != artifact_record:
        raise RuntimeError(
            "A different Task 2 artifact-verification record already exists."
        )
else:
    atomic_write_json(artifact_record, artifact_record_path)

print(json.dumps(artifact_record, indent=2))
print(f"\nPersisted Block 01 record: {persistent_preflight}")
print(f"Saved artifact verification: {artifact_record_path}")
print("PACS traversed: False")
print("Sketch images accessed: 0")
print("Next action: inspect this record before preparing source-only code and data.")
