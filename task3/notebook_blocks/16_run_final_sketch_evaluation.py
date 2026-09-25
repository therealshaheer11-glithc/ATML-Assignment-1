"""Block 16: one-time final Sketch evaluation after exact-lock authorization."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

import torch
from PIL import Image


EXPECTED_PROTOCOL_SHA256 = "e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74"
EXPECTED_ARCHIVE_SHA256 = "0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102"
EXPECTED_TASK2_RESULTS_SHA256 = "aa3685b1d5370e20d76798fb2ebbc7cb2286e7466c2dfdd8a4414799aad65883"
EXPECTED_TASK2_PREDICTIONS_SHA256 = "35c6fb24a9f7fb2b576bfd9dbae84eb2c0ab2bbbf4e71e370f8d512a9c913bb3"
EXPECTED_ARCHIVE_MEMBERS = 10044
EXPECTED_TARGET_IMAGES = 3929
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

CODE_ROOT = Path("/content/atml_pa1_task3_source")
ATML_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_ROOT = ATML_ROOT / "task3_domain_generalization_20260924"
PROVENANCE = TASK3_ROOT / "provenance"
FINAL_LOCK = PROVENANCE / "final_experiment_lock.json"
LOCK_COMPLETION = PROVENANCE / "final_experiment_lock_completion.json"
AUTHORIZATION = PROVENANCE / "final_sketch_evaluation_authorization.json"
COMPLETION = PROVENANCE / "final_sketch_evaluation_completion.json"
PARENT_PROTOCOL = CODE_ROOT / "shared" / "splits" / "pacs_sketch_seed6304.json"
PACS_ARCHIVE = ATML_ROOT / "datasets" / "PACS_dassl.zip"
TASK2_FINAL_ROOT = ATML_ROOT / "task2_final_evaluation_20260923"
TASK2_RESULTS = TASK2_FINAL_ROOT / "final_results.json"
TASK2_PREDICTIONS = TASK2_FINAL_ROOT / "target_predictions.csv"
SKETCH_ROOT = Path("/content/task3_pacs_sketch_final_v1")
PARTIAL_SKETCH_ROOT = Path("/content/task3_pacs_sketch_final_v1.partial")
FINAL_OUTPUT = TASK3_ROOT / "final_evaluation"
PARTIAL_OUTPUT = TASK3_ROOT / "final_evaluation.partial"


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def archive_target_path(member_name: str, expected_paths: set[str]) -> str | None:
    path = PurePosixPath(member_name)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe archive member: {member_name}")
    positions = [
        index for index, part in enumerate(path.parts) if part.lower() == "sketch"
    ]
    if not positions:
        return None
    if len(positions) != 1:
        raise RuntimeError(f"Ambiguous Sketch archive member: {member_name}")
    relative = PurePosixPath(*path.parts[positions[0] :]).as_posix()
    return relative if relative in expected_paths else None


def verify_sketch_workspace(root: Path, expected_paths: set[str]) -> str:
    actual: set[str] = set()
    digest = hashlib.sha256()
    unreadable: list[str] = []
    for path in sorted((root / "sketch").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        actual.add(relative)
        digest.update(f"{relative}\t{path.stat().st_size}\n".encode())
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            unreadable.append(relative)
    if unreadable:
        raise RuntimeError(f"Unreadable Sketch images: {unreadable[:5]}")
    if actual != expected_paths:
        raise RuntimeError(
            "Final Sketch workspace differs from the locked target records. "
            f"Missing={sorted(expected_paths - actual)[:5]}, "
            f"unexpected={sorted(actual - expected_paths)[:5]}"
        )
    return digest.hexdigest()


# Gate 1: exact reviewed lock hash supplied by the user after Block 15.
approved_lock_sha256 = os.environ.get("TASK3_APPROVED_FINAL_LOCK_SHA256", "").strip()
if len(approved_lock_sha256) != 64:
    raise RuntimeError(
        "Set TASK3_APPROVED_FINAL_LOCK_SHA256 to the reviewed Block 15 lock hash"
    )
actual_lock_sha256 = sha256_file(FINAL_LOCK)
if actual_lock_sha256 != approved_lock_sha256:
    raise RuntimeError(
        f"Reviewed lock hash differs: {actual_lock_sha256} != {approved_lock_sha256}"
    )
lock = load_json(FINAL_LOCK, "final experiment lock")
lock["lock_sha256"] = actual_lock_sha256
for name, expected in {
    "status": "TASK3_FINAL_EXPERIMENT_LOCKED_BEFORE_TARGET_ACCESS",
    "approved_after_complete_source_diagnostics": True,
    "further_tuning_permitted": False,
    "checkpoint_replacement_permitted": False,
    "target_labels_accessed": False,
    "sketch_images_accessed": 0,
    "final_evaluation_started": False,
}.items():
    if lock.get(name) != expected:
        raise RuntimeError(f"Final lock field differs: {name}")
lock_completion = load_json(LOCK_COMPLETION, "Block 15 completion")
if lock_completion.get("status") != "TASK3_BLOCK_15_FINAL_EXPERIMENT_LOCK_PASS":
    raise RuntimeError("Block 15 did not pass")
if lock_completion.get("final_lock", {}).get("sha256") != actual_lock_sha256:
    raise RuntimeError("Block 15 completion points to a different lock")

# Gate 2: code, repository, outputs, and all frozen checkpoint identities are intact.
if not (CODE_ROOT / ".git").is_dir():
    raise FileNotFoundError(f"Repository checkout is missing: {CODE_ROOT}")
if subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True
).strip():
    raise RuntimeError("The final-evaluation checkout is not clean")
current_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
).strip()
if current_commit != lock["repository_commit"]:
    raise RuntimeError("Repository commit differs from the final lock")
sys.path.insert(0, str(CODE_ROOT))
from task3.config import code_tree_sha256  # noqa: E402
from task3.evaluation.final_sketch import (  # noqa: E402
    MODEL_ORDER,
    run_final_evaluation,
    runtime_identity,
    validate_target_records,
)

if code_tree_sha256(CODE_ROOT) != lock["code_tree_sha256"]:
    raise RuntimeError("Code tree differs from the final lock")
for model_name in MODEL_ORDER:
    record = lock["checkpoints"][model_name]
    if sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint changed after lock: {model_name}")
if FINAL_OUTPUT.exists() or PARTIAL_OUTPUT.exists() or COMPLETION.exists():
    raise FileExistsError("Final evaluation already exists or has a partial output")
if not torch.cuda.is_available():
    raise RuntimeError("The locked final evaluation requires CUDA")
device = torch.device("cuda")
if runtime_identity(device) != EXPECTED_RUNTIME:
    raise RuntimeError(f"Final runtime differs: {runtime_identity(device)}")
if lock["final_evaluation_rules"].get("runtime") != EXPECTED_RUNTIME:
    raise RuntimeError("Final lock contains a different runtime")

# Gate 3: authenticate target-bearing files before parsing any of them.
if sha256_file(PARENT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError("Locked PACS protocol changed")
if sha256_file(PACS_ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
    raise RuntimeError("Locked PACS archive changed")
if sha256_file(TASK2_RESULTS) != EXPECTED_TASK2_RESULTS_SHA256:
    raise RuntimeError("Task 2 final results changed")
if sha256_file(TASK2_PREDICTIONS) != EXPECTED_TASK2_PREDICTIONS_SHA256:
    raise RuntimeError("Task 2 target predictions changed")

authorization = {
    "status": "TASK3_FINAL_SKETCH_EVALUATION_AUTHORIZED",
    "reviewed_final_lock_sha256": actual_lock_sha256,
    "repository_commit": current_commit,
    "code_tree_sha256": lock["code_tree_sha256"],
    "single_complete_target_evaluation": True,
    "post_target_training_or_selection": False,
}
if AUTHORIZATION.exists():
    if load_json(AUTHORIZATION, "final authorization") != authorization:
        raise RuntimeError("Existing final authorization differs")
else:
    atomic_write_json(authorization, AUTHORIZATION)

# Target access begins only here, after every lock and identity gate above passed.
protocol = load_json(PARENT_PROTOCOL, "locked PACS protocol")
for name, expected in {
    "dataset": "PACS",
    "seed": 6304,
    "target": "sketch",
    "classes": ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
    "file_list_sha256": "559ac63b8df8e07330b97112e5ec4c414b3957585b28d21b4cfecd2181f538e0",
}.items():
    if protocol.get(name) != expected:
        raise RuntimeError(f"Locked target protocol field differs: {name}")
target_records = protocol.get("target_unlabeled", [])
validate_target_records(target_records)
expected_paths = {record["path"] for record in target_records}

if SKETCH_ROOT.exists():
    sentinel = SKETCH_ROOT / "TARGET_SNAPSHOT.json"
    if not sentinel.is_file():
        raise RuntimeError("Existing final Sketch workspace lacks its sentinel")
    snapshot_sha256 = verify_sketch_workspace(SKETCH_ROOT, expected_paths)
    saved = load_json(sentinel, "target snapshot sentinel")
    if saved.get("snapshot_sha256") != snapshot_sha256:
        raise RuntimeError("Existing final Sketch snapshot changed")
    reused_workspace = True
else:
    if PARTIAL_SKETCH_ROOT.exists():
        raise RuntimeError("A partial Sketch extraction exists; inspect before retrying")
    PARTIAL_SKETCH_ROOT.mkdir(parents=True)
    extracted: set[str] = set()
    with zipfile.ZipFile(PACS_ARCHIVE) as archive:
        members = archive.infolist()
        if len(members) != EXPECTED_ARCHIVE_MEMBERS:
            raise RuntimeError("PACS archive member count differs")
        for member in members:
            if member.is_dir():
                continue
            relative = archive_target_path(member.filename, expected_paths)
            if relative is None:
                continue
            if relative in extracted:
                raise RuntimeError(f"Duplicate target archive member: {relative}")
            destination = (PARTIAL_SKETCH_ROOT / relative).resolve()
            if PARTIAL_SKETCH_ROOT.resolve() not in destination.parents:
                raise RuntimeError("Unsafe target extraction destination")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.add(relative)
    if extracted != expected_paths or len(extracted) != EXPECTED_TARGET_IMAGES:
        raise RuntimeError("Archive extraction did not reproduce all target records")
    snapshot_sha256 = verify_sketch_workspace(PARTIAL_SKETCH_ROOT, expected_paths)
    atomic_write_json(
        {
            "status": "TASK3_FINAL_TARGET_SNAPSHOT_VERIFIED",
            "lock_sha256": actual_lock_sha256,
            "target_count": EXPECTED_TARGET_IMAGES,
            "snapshot_sha256": snapshot_sha256,
        },
        PARTIAL_SKETCH_ROOT / "TARGET_SNAPSHOT.json",
    )
    os.replace(PARTIAL_SKETCH_ROOT, SKETCH_ROOT)
    reused_workspace = False

task2_results = load_json(TASK2_RESULTS, "Task 2 final results")
with TASK2_PREDICTIONS.open(newline="") as handle:
    task2_predictions = list(csv.DictReader(handle))
if len(task2_predictions) != EXPECTED_TARGET_IMAGES:
    raise RuntimeError("Task 2 prediction count differs")

print("All Block 16 post-lock gates passed.")
print("Final lock:", actual_lock_sha256)
print("All eight checkpoints will be evaluated exactly once on 3,929 Sketch images.")
print("No target result can change training, selection, or checkpoint identity.\n")

final = run_final_evaluation(
    lock=lock,
    sketch_root=SKETCH_ROOT,
    target_records=target_records,
    output_dir=PARTIAL_OUTPUT,
    device=device,
    task2_results=task2_results,
    task2_results_identity={"path": str(TASK2_RESULTS), "sha256": EXPECTED_TASK2_RESULTS_SHA256},
    task2_predictions=task2_predictions,
    task2_predictions_identity={"path": str(TASK2_PREDICTIONS), "sha256": EXPECTED_TASK2_PREDICTIONS_SHA256},
)
os.replace(PARTIAL_OUTPUT, FINAL_OUTPUT)

artifacts = []
for path in sorted(FINAL_OUTPUT.rglob("*")):
    if path.is_file():
        artifacts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
completion = {
    "status": "TASK3_BLOCK_16_FINAL_SKETCH_EVALUATION_PASS",
    "experiment_lock_sha256": actual_lock_sha256,
    "repository_commit": current_commit,
    "code_tree_sha256": lock["code_tree_sha256"],
    "target_count": EXPECTED_TARGET_IMAGES,
    "sketch_images_accessed": EXPECTED_TARGET_IMAGES,
    "target_labels_used_for_training_or_selection": False,
    "all_eight_checkpoints_evaluated": True,
    "main_comparison": ["erm", "dan_dg_1", "sam"],
    "target_workspace": {
        "path": str(SKETCH_ROOT),
        "snapshot_sha256": snapshot_sha256,
        "reused": reused_workspace,
    },
    "runtime": runtime_identity(device),
    "artifacts": artifacts,
    "summary": {
        name: {
            "accuracy": result["target"]["accuracy"],
            "macro_f1": result["target"]["macro_f1"],
            "delta_accuracy_vs_erm": result["target"]["delta_accuracy_vs_erm"],
        }
        for name, result in final["results"].items()
    },
}
atomic_write_json(completion, COMPLETION)

print("Final evaluation completed. No checkpoint was changed.\n")
print(json.dumps(completion, indent=2))
print("\nSaved at:", COMPLETION)
