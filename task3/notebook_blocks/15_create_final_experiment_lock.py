"""Block 15: authenticate all source-only evidence and create the final lock.

This block deliberately does not parse target records, list archive members, open
Sketch images, or read Task 2 target-bearing result files.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DIAGNOSTIC_COMMIT = "136e28b556d38c99e486e2d53ddff33d921fbfc6"
EXPECTED_FINAL_CODE_SHA256 = (
    "e5d2bf6341c341d532aeccbae89a34dca01ff3b7dc2e8dfddc782d04b58f5b05"
)
EXPECTED_ALL_MODELS_SHA256 = (
    "578253da8a890d9675996ed0f9f45a8f5744836e7c64aa43e1cb690ff64530dc"
)
EXPECTED_PROTOCOL_SHA256 = (
    "e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74"
)
EXPECTED_ARCHIVE_SHA256 = (
    "0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102"
)
EXPECTED_TASK2_RESULTS_SHA256 = (
    "aa3685b1d5370e20d76798fb2ebbc7cb2286e7466c2dfdd8a4414799aad65883"
)
EXPECTED_TASK2_PREDICTIONS_SHA256 = (
    "35c6fb24a9f7fb2b576bfd9dbae84eb2c0ab2bbbf4e71e370f8d512a9c913bb3"
)
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

REQUIRED_FINAL_UPDATE = {
    "task3/docs/TASK3_FINAL_EVALUATION_PROTOCOL.md",
    "task3/evaluation/final_sketch.py",
    "task3/notebook_blocks/15_create_final_experiment_lock.py",
    "task3/notebook_blocks/16_run_final_sketch_evaluation.py",
    "task3/provenance/RUN_LOG.md",
    "task3/tests/test_task3_final_sketch.py",
}

CODE_ROOT = Path("/content/atml_pa1_task3_source")
ATML_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_ROOT = ATML_ROOT / "task3_domain_generalization_20260924"
PROVENANCE = TASK3_ROOT / "provenance"
PRIMARY_TRAINING = TASK3_ROOT / "training"
VARIANT_ROOT = (
    TASK3_ROOT
    / "research_variants"
    / "dan_dg_initial_bandwidth_floor_v1"
)
VARIANT_TRAINING = VARIANT_ROOT / "training"
VARIANT_PROVENANCE = VARIANT_ROOT / "provenance"
ALL_MODELS_RESULT = (
    VARIANT_ROOT / "source_diagnostics" / "all_models_source_diagnostics.json"
)
BLOCK14_COMPLETION = VARIANT_PROVENANCE / "source_diagnostics_completion.json"
TASK2_ROOT = ATML_ROOT / "task2_corrected_normalized_v3_20260923"
PARENT_PROTOCOL = CODE_ROOT / "shared" / "splits" / "pacs_sketch_seed6304.json"
PACS_ARCHIVE = ATML_ROOT / "datasets" / "PACS_dassl.zip"
TASK2_FINAL_ROOT = ATML_ROOT / "task2_final_evaluation_20260923"
FINAL_ROOT = TASK3_ROOT / "final_evaluation"
FINAL_LOCK = PROVENANCE / "final_experiment_lock.json"
LOCK_COMPLETION = PROVENANCE / "final_experiment_lock_completion.json"

MODEL_ORDER = (
    "erm",
    "dan_dg_1",
    "sam",
    "dan_dg_0p1",
    "dan_dg_10",
    "dan_dg_floor_0p1",
    "dan_dg_floor_1",
    "dan_dg_floor_10",
)
SPECS = {
    "erm": (TASK2_ROOT / "source_only" / "best.pt", "3d28a223e4b97b323cb3a20dcb5b7577af96631f2e6ef1f2bc99d53d85761327", 4),
    "dan_dg_1": (PRIMARY_TRAINING / "dan_dg_1" / "best.pt", "a44bff8e519134459c2d6bf056785801f9944945d160e516b4da7f5e2752992b", 3),
    "sam": (PRIMARY_TRAINING / "sam" / "best.pt", "040a80ce15974d71f3a90b5aab5050965d27212822079e5a075c742052c0827d", 5),
    "dan_dg_0p1": (PRIMARY_TRAINING / "dan_dg_0p1" / "best.pt", "dc6036a28e3af8c281b143adb6f47b4d03676fa44c117e912ef9bcd959d8ca27", 5),
    "dan_dg_10": (PRIMARY_TRAINING / "dan_dg_10" / "best.pt", "f8cc723dc16b5e17a48f8454541beb473b70e6c38d45895f170c09f59821b2c8", 7),
    "dan_dg_floor_0p1": (VARIANT_TRAINING / "dan_dg_floor_0p1" / "best.pt", "46e770107eb563517239d524622bc7cb121d9e166de398b692350fc80c0a453c", 25),
    "dan_dg_floor_1": (VARIANT_TRAINING / "dan_dg_floor_1" / "best.pt", "6b8c8ac0f373882b468e2fbf98bfeb4f97e1a799bf073cc6a7f3667d9747418a", 7),
    "dan_dg_floor_10": (VARIANT_TRAINING / "dan_dg_floor_10" / "best.pt", "030adaa198612ccda2914aa283c83c12554f16a210e82ba5f4e3fcd93c4d5483", 17),
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


# Gate 1: clean, exact, committed final-evaluation implementation update.
if not (CODE_ROOT / ".git").is_dir():
    raise FileNotFoundError(f"Repository checkout is missing: {CODE_ROOT}")
dirty = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True
).strip()
if dirty:
    raise RuntimeError("The Colab checkout has uncommitted changes:\n" + dirty)
subprocess.run(["git", "pull", "--ff-only"], cwd=CODE_ROOT, check=True)
final_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
).strip()
ancestor = subprocess.run(
    ["git", "merge-base", "--is-ancestor", DIAGNOSTIC_COMMIT, final_commit],
    cwd=CODE_ROOT,
)
if ancestor.returncode != 0 or final_commit == DIAGNOSTIC_COMMIT:
    raise RuntimeError("Final-evaluation commit is not a strict diagnostic descendant")
changed = set(
    filter(
        None,
        subprocess.check_output(
            ["git", "diff", "--name-only", f"{DIAGNOSTIC_COMMIT}..{final_commit}"],
            cwd=CODE_ROOT,
            text=True,
        ).splitlines(),
    )
)
if changed != REQUIRED_FINAL_UPDATE:
    raise RuntimeError(
        "Final-evaluation Git update differs from the approved file set. "
        f"Missing={sorted(REQUIRED_FINAL_UPDATE - changed)}, "
        f"unexpected={sorted(changed - REQUIRED_FINAL_UPDATE)}"
    )

sys.path.insert(0, str(CODE_ROOT))
from task3.config import code_tree_sha256  # noqa: E402

actual_code_sha256 = code_tree_sha256(CODE_ROOT)
if actual_code_sha256 != EXPECTED_FINAL_CODE_SHA256:
    raise RuntimeError(
        f"Final code tree differs: {actual_code_sha256} != {EXPECTED_FINAL_CODE_SHA256}"
    )
tests = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "task3/tests", "-v"],
    cwd=CODE_ROOT,
)
if tests.returncode != 0:
    raise RuntimeError("The complete target-free Task 3 test suite failed")

# Gate 2: authenticate the complete, pre-lock, source-only diagnostic record.
completion = load_json(BLOCK14_COMPLETION, "Block 14 completion")
for name, expected in {
    "status": "TASK3_BLOCK_14_ALL_SOURCE_DIAGNOSTICS_COVERAGE_PASS",
    "source_only_phase": True,
    "diagnostics_completed": True,
    "experiment_lock_created": False,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "diagnostic_commit": DIAGNOSTIC_COMMIT,
    "diagnostic_code_tree_sha256": "2bf8c537ed21fe2eb64f5ec122ba94dbc37aef615395f988a1c4d44c680bdc39",
}.items():
    if completion.get(name) != expected:
        raise RuntimeError(f"Block 14 completion field differs: {name}")
if sha256_file(ALL_MODELS_RESULT) != EXPECTED_ALL_MODELS_SHA256:
    raise RuntimeError("The unified source-diagnostic result changed")
all_models = load_json(ALL_MODELS_RESULT, "unified source diagnostics")
if tuple(all_models.get("models", {})) != (
    "erm", "dan_dg_0p1", "dan_dg_1", "dan_dg_10", "sam",
    "dan_dg_floor_0p1", "dan_dg_floor_1", "dan_dg_floor_10",
):
    raise RuntimeError("Unified source diagnostics have a different model set or order")

# Gate 3: authenticate every selected checkpoint without opening any target evidence.
checkpoints: dict[str, dict] = {}
for model_name in MODEL_ORDER:
    path, expected_hash, epoch = SPECS[model_name]
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RuntimeError(f"Selected checkpoint identity differs: {model_name}")
    diagnostic = all_models["models"][model_name]
    checkpoints[model_name] = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": expected_hash,
        "selected_epoch": epoch,
        "source_validation": diagnostic["source_validation_reproduced"],
        "source_diagnostics": {
            "source_domain_separability": diagnostic["source_domain_separability"],
            "common_sharpness_proxy": diagnostic["common_sharpness_proxy"],
        },
        "research_variant": model_name.startswith("dan_dg_floor_"),
    }

# Gate 4: hash only the opaque protocol and archive files. Do not parse, list, or open
# target-bearing content in this pre-lock block.
if sha256_file(PARENT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError("The locked Task 2/PACS protocol file changed")
if sha256_file(PACS_ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
    raise RuntimeError("The locked PACS archive changed")
if FINAL_LOCK.exists() or LOCK_COMPLETION.exists() or FINAL_ROOT.exists():
    raise FileExistsError("A final lock or final-evaluation output already exists")

lock = {
    "status": "TASK3_FINAL_EXPERIMENT_LOCKED_BEFORE_TARGET_ACCESS",
    "protocol_version": "task3-approved-2026-09-24-v1",
    "lock_version": "task3-final-lock-2026-09-25-v1",
    "approved_after_complete_source_diagnostics": True,
    "further_tuning_permitted": False,
    "checkpoint_replacement_permitted": False,
    "target_labels_accessed": False,
    "sketch_images_accessed": 0,
    "final_evaluation_started": False,
    "repository_commit": final_commit,
    "code_tree_sha256": actual_code_sha256,
    "parent_protocol": {
        "path": str(PARENT_PROTOCOL),
        "sha256": EXPECTED_PROTOCOL_SHA256,
        "target_record_count_after_lock": 3929,
    },
    "pacs_archive": {
        "path": str(PACS_ARCHIVE),
        "sha256": EXPECTED_ARCHIVE_SHA256,
        "members_not_listed_by_this_block": True,
    },
    "source_diagnostics": {
        "path": str(ALL_MODELS_RESULT),
        "sha256": EXPECTED_ALL_MODELS_SHA256,
    },
    "model_order": list(MODEL_ORDER),
    "main_comparison": ["erm", "dan_dg_1", "sam"],
    "original_strength_study": ["erm", "dan_dg_0p1", "dan_dg_1", "dan_dg_10"],
    "supplementary_floor_study": [
        "erm", "dan_dg_floor_0p1", "dan_dg_floor_1", "dan_dg_floor_10"
    ],
    "checkpoints": checkpoints,
    "final_evaluation_rules": {
        "single_complete_target_evaluation": True,
        "target_count": 3929,
        "transform": "resize_256_center_crop_224_imagenet_normalization",
        "batch_size": 64,
        "shuffle": False,
        "metrics": [
            "accuracy", "macro_f1_all_7_classes", "per_class_accuracy",
            "confusion_matrix", "delta_accuracy_vs_erm",
        ],
        "class_change_reference": "erm",
        "class_tie_policy": "first_in_fixed_pacs_class_order",
        "example_order": "lexicographic_image_path",
        "examples_per_direction_per_model": 3,
        "post_target_training_or_selection": False,
        "runtime": EXPECTED_RUNTIME,
    },
    "task2_comparison_evidence_deferred_until_after_lock": {
        "final_results_path": str(TASK2_FINAL_ROOT / "final_results.json"),
        "final_results_sha256": EXPECTED_TASK2_RESULTS_SHA256,
        "target_predictions_path": str(TASK2_FINAL_ROOT / "target_predictions.csv"),
        "target_predictions_sha256": EXPECTED_TASK2_PREDICTIONS_SHA256,
        "files_read_by_this_block": False,
    },
}
atomic_write_json(lock, FINAL_LOCK)
lock_sha256 = sha256_file(FINAL_LOCK)
lock_completion = {
    "status": "TASK3_BLOCK_15_FINAL_EXPERIMENT_LOCK_PASS",
    "target_free_lock_creation": True,
    "target_labels_accessed": False,
    "sketch_images_accessed": 0,
    "final_evaluation_started": False,
    "repository_commit": final_commit,
    "code_tree_sha256": actual_code_sha256,
    "target_free_tests_passed": True,
    "final_lock": {"path": str(FINAL_LOCK), "sha256": lock_sha256},
    "review_required_before_block_16": True,
}
atomic_write_json(lock_completion, LOCK_COMPLETION)

print("All Block 15 target-free gates passed.")
print("All eight selected checkpoints are now immutable.")
print("Sketch records parsed: 0")
print("Sketch images accessed: 0")
print("Task 2 target-result files read: 0")
print("Final evaluation started: False")
print("\nAudited final-lock completion:\n")
print(json.dumps(lock_completion, indent=2))
print("\nReview and send this complete output before Block 16.")
