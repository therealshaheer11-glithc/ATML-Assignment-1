"""Block 14: complete source diagnostics for every Task 3 checkpoint."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


TRAINING_COMMIT = "dc3acfdf547e7bc29bd381b3fe05e271879f18d0"
TRAINING_CODE_SHA256 = (
    "34d2778e76a5a7522a9c2eac2682eb4d2a09ee9de0e88b6d04d7d0a68e26ce13"
)
PRIMARY_TRAINING_COMMIT = "19208b4c62acb980fb3246f30e062784b90d8dfc"
PRIMARY_TRAINING_CODE_SHA256 = (
    "4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44"
)
EXPECTED_DIAGNOSTIC_CODE_SHA256 = (
    "2bf8c537ed21fe2eb64f5ec122ba94dbc37aef615395f988a1c4d44c680bdc39"
)
VARIANT_ID = "dan_dg_initial_bandwidth_floor_v1"
VARIANT_PROTOCOL_VERSION = "task3-research-bandwidth-floor-2026-09-25-v1"

REQUIRED_DIAGNOSTIC_UPDATE = {
    "task3/docs/DAN_DG_BANDWIDTH_FLOOR_RESEARCH_VARIANT.md",
    "task3/notebook_blocks/14_run_bandwidth_floor_source_diagnostics.py",
    "task3/provenance/RUN_LOG.md",
    "task3/research_variants/run_source_diagnostics.py",
    "task3/tests/test_task3_bandwidth_floor_diagnostics.py",
}

CODE_ROOT = Path("/content/atml_pa1_task3_source")
SOURCE_ROOT = Path("/content/task3_pacs_sources_v1")
ATML_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_ROOT = ATML_ROOT / "task3_domain_generalization_20260924"
SOURCE_PROTOCOL = TASK3_ROOT / "source_protocol" / "pacs_sources_seed6304.json"
PRIMARY_PROVENANCE = TASK3_ROOT / "provenance"
PRIMARY_DIAGNOSTICS = TASK3_ROOT / "source_diagnostics"
PRIMARY_TRAINING = TASK3_ROOT / "training"
PRIMARY_DIAGNOSTIC_RESULT = PRIMARY_DIAGNOSTICS / "source_diagnostics.json"
VARIANT_ROOT = TASK3_ROOT / "research_variants" / VARIANT_ID
VARIANT_PROVENANCE = VARIANT_ROOT / "provenance"
DIAGNOSTICS_ROOT = VARIANT_ROOT / "source_diagnostics"
DIAGNOSTICS_RESULT = DIAGNOSTICS_ROOT / "source_diagnostics.json"
ALL_MODELS_RESULT = DIAGNOSTICS_ROOT / "all_models_source_diagnostics.json"
AUTHORIZATION = VARIANT_PROVENANCE / "source_diagnostics_authorization.json"
COMPLETION = VARIANT_PROVENANCE / "source_diagnostics_completion.json"

PRIMARY_RUN_COMPLETIONS = {
    "dan_dg_0p1": {
        "status": "TASK3_BLOCK_06_DAN_DG_0P1_PASS",
        "mmd_lambda": 0.1,
        "checkpoint": (
            "dc6036a28e3af8c281b143adb6f47b4d03676fa44c117e912ef9bcd959d8ca27"
        ),
        "history": (
            "fc92efd81ea529a48373977f9d4973e754eb53cfca7622c8513ad5cfc64b35b8"
        ),
        "manifest": (
            "731f07bded36f6034e014eb217dca2b1b3a9c7836ed0e5bd72d0c2be3537672a"
        ),
    },
    "dan_dg_10": {
        "status": "TASK3_BLOCK_07_DAN_DG_10_PASS",
        "mmd_lambda": 10.0,
        "checkpoint": (
            "f8cc723dc16b5e17a48f8454541beb473b70e6c38d45895f170c09f59821b2c8"
        ),
        "history": (
            "dec8773f2a0642b2aecea85619dbd8bdc19a74723c06a5ce9ca24ce9c8b4b918"
        ),
        "manifest": (
            "962626d182952f068a689bb213474074f8f07869c9a5a254cc246164d8fe4330"
        ),
    },
}

RUN_COMPLETIONS = {
    "dan_dg_floor_1": {
        "status": "TASK3_BLOCK_11_DAN_DG_FLOOR_1_PASS",
        "mmd_lambda": 1.0,
        "checkpoint": (
            "6b8c8ac0f373882b468e2fbf98bfeb4f97e1a799bf073cc6a7f3667d9747418a"
        ),
        "history": (
            "321e441f428198a3479c3a375532ae87ccd8aeabc4c0584330b6bd8cc7af98cf"
        ),
        "manifest": (
            "66c9dcf2d866a0abddc1f2a85b913fc0de20ac0c833af135ec13c4cc19e82bf6"
        ),
    },
    "dan_dg_floor_0p1": {
        "status": "TASK3_BLOCK_12_DAN_DG_FLOOR_0P1_PASS",
        "mmd_lambda": 0.1,
        "checkpoint": (
            "46e770107eb563517239d524622bc7cb121d9e166de398b692350fc80c0a453c"
        ),
        "history": (
            "50246e91eeb0d53eabb1cd8fc0901af6b5c00b23918b2aaaf8a44b333ac0617f"
        ),
        "manifest": (
            "ff7fccf71ab28d7e4fe5b22b4bfdfa5b721e2eedaaf3d9809f98900d91404242"
        ),
    },
    "dan_dg_floor_10": {
        "status": "TASK3_BLOCK_13_DAN_DG_FLOOR_10_PASS",
        "mmd_lambda": 10.0,
        "checkpoint": (
            "030adaa198612ccda2914aa283c83c12554f16a210e82ba5f4e3fcd93c4d5483"
        ),
        "history": (
            "1597de06e3976f9f80eac598f0f8c9e08b5d165d4be3f6cad26121c18866dbf6"
        ),
        "manifest": (
            "db388ebdd098b4cf89ffe4ecf44a7cb372b3aaa381e21116de221ea0d79ece31"
        ),
    },
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


def verify_artifact(entry: dict, expected: str, description: str) -> None:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise RuntimeError(f"Completion record lacks {description}")
    path = Path(entry["path"])
    if entry.get("sha256") != expected:
        raise RuntimeError(f"Recorded {description} hash differs")
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"Stored {description} identity differs")


# Gate 1: exact committed diagnostic-only update after immutable variant training.
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
ancestor = subprocess.run(
    ["git", "merge-base", "--is-ancestor", TRAINING_COMMIT, diagnostic_commit],
    cwd=CODE_ROOT,
)
if ancestor.returncode != 0 or diagnostic_commit == TRAINING_COMMIT:
    raise RuntimeError("Diagnostic commit is not a strict training-commit descendant")
changed = set(
    filter(
        None,
        subprocess.check_output(
            ["git", "diff", "--name-only", f"{TRAINING_COMMIT}..{diagnostic_commit}"],
            cwd=CODE_ROOT,
            text=True,
        ).splitlines(),
    )
)
missing = REQUIRED_DIAGNOSTIC_UPDATE - changed
unexpected = changed - REQUIRED_DIAGNOSTIC_UPDATE
if missing or unexpected:
    raise RuntimeError(
        "Variant diagnostic Git update differs from the approved file set. "
        f"Missing={sorted(missing)}, unexpected={sorted(unexpected)}"
    )

# Gate 2: exact implementation plus the complete 29-test target-free suite.
sys.path.insert(0, str(CODE_ROOT))
from task3.config import code_tree_sha256  # noqa: E402

actual_code_sha256 = code_tree_sha256(CODE_ROOT)
if actual_code_sha256 != EXPECTED_DIAGNOSTIC_CODE_SHA256:
    raise RuntimeError(
        f"Variant diagnostic code tree differs: {actual_code_sha256} "
        f"!= {EXPECTED_DIAGNOSTIC_CODE_SHA256}"
    )
test_result = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "task3/tests", "-v"],
    cwd=CODE_ROOT,
)
if test_result.returncode != 0:
    raise RuntimeError("The 29-test target-free suite failed")

# Gate 3: authenticate the two primary strength-study runs not covered by Block 09.
for run_id, specification in PRIMARY_RUN_COMPLETIONS.items():
    record_path = PRIMARY_PROVENANCE / f"{run_id}_training_completion.json"
    record = load_json(record_path, f"{run_id} completion")
    expected = {
        "status": specification["status"],
        "run_id": run_id,
        "method": "dan_dg",
        "mmd_lambda": specification["mmd_lambda"],
        "source_only_phase": True,
        "training_completed": True,
        "sketch_images_accessed": 0,
        "repository_commit": PRIMARY_TRAINING_COMMIT,
        "code_tree_sha256": PRIMARY_TRAINING_CODE_SHA256,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise RuntimeError(f"{run_id} completion field differs: {name}")
    verify_artifact(
        record.get("best_checkpoint"),
        specification["checkpoint"],
        f"{run_id} checkpoint",
    )
    verify_artifact(
        record.get("history"), specification["history"], f"{run_id} history"
    )
    verify_artifact(
        record.get("run_manifest"),
        specification["manifest"],
        f"{run_id} manifest",
    )

# Gate 4: authenticate every completed research run and immutable artifact.
for run_id, specification in RUN_COMPLETIONS.items():
    record_path = VARIANT_PROVENANCE / f"{run_id}_training_completion.json"
    record = load_json(record_path, f"{run_id} completion")
    expected = {
        "status": specification["status"],
        "variant_id": VARIANT_ID,
        "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
        "run_id": run_id,
        "method": "dan_dg_bandwidth_floor",
        "mmd_lambda": specification["mmd_lambda"],
        "research_variant": True,
        "primary_protocol_replacement": False,
        "source_only_phase": True,
        "training_completed": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
        "repository_commit": TRAINING_COMMIT,
        "code_tree_sha256": TRAINING_CODE_SHA256,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise RuntimeError(f"{run_id} completion field differs: {name}")
    verify_artifact(record.get("best_checkpoint"), specification["checkpoint"], f"{run_id} checkpoint")
    verify_artifact(record.get("history"), specification["history"], f"{run_id} history")
    verify_artifact(record.get("run_manifest"), specification["manifest"], f"{run_id} manifest")

# Gate 5: reuse, rather than redraw, both approved primary diagnostic subsets.
primary_completion = load_json(
    PRIMARY_PROVENANCE / "source_diagnostics_completion.json",
    "primary source-diagnostic completion",
)
for name, value in {
    "status": "TASK3_BLOCK_09_SOURCE_DIAGNOSTICS_PASS",
    "source_only_phase": True,
    "diagnostics_completed": True,
    "experiment_lock_created": False,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
}.items():
    if primary_completion.get(name) != value:
        raise RuntimeError(f"Primary source-diagnostic field differs: {name}")
for name, expected in {
    "probe_design": "6f94666bd693328a6cdf6b2b1193a97267f7367b3ff62981c38a62651adc8f3c",
    "sharpness_batch_design": "ac5ed81e14e7867dc616416de9da656e56d5c9fc88e02fe6ef66aa2ffffc7cab",
}.items():
    entry = primary_completion.get(name, {})
    verify_artifact(entry, expected, f"primary {name}")
verify_artifact(
    primary_completion.get("result"),
    "cdf83aa3fb41128f39a89c3358eeb7a7b09288862c98274473bb7aeb41830e1f",
    "primary source-diagnostic result",
)

if COMPLETION.exists() or DIAGNOSTICS_RESULT.exists() or ALL_MODELS_RESULT.exists():
    raise FileExistsError("Variant source diagnostics already have outputs; do not overwrite")

authorization = {
    "status": "TASK3_ALL_SOURCE_DIAGNOSTICS_COVERAGE_AUTHORIZED",
    "variant_id": VARIANT_ID,
    "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
    "research_variant": True,
    "primary_protocol_replacement": False,
    "approved_diagnostics": [
        "source_domain_separability",
        "common_local_sharpness_proxy",
    ],
    "previously_completed_models_reused": ["erm", "dan_dg_1", "sam"],
    "newly_evaluated_models": [
        *PRIMARY_RUN_COMPLETIONS,
        *RUN_COMPLETIONS,
    ],
    "complete_model_coverage": [
        "erm",
        "dan_dg_0p1",
        "dan_dg_1",
        "dan_dg_10",
        "sam",
        "dan_dg_floor_0p1",
        "dan_dg_floor_1",
        "dan_dg_floor_10",
    ],
    "subset_policy": "reuse_exact_primary_Block_09_designs_and_partition",
    "source_only_phase": True,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "training_commit": TRAINING_COMMIT,
    "training_code_tree_sha256": TRAINING_CODE_SHA256,
    "diagnostic_commit": diagnostic_commit,
    "diagnostic_code_tree_sha256": actual_code_sha256,
}
if AUTHORIZATION.exists():
    if load_json(AUTHORIZATION, "variant source-diagnostic authorization") != authorization:
        raise RuntimeError("Existing variant source-diagnostic authorization differs")
else:
    atomic_write_json(authorization, AUTHORIZATION)

command = [
    sys.executable,
    "-m",
    "task3.research_variants.run_source_diagnostics",
    "--code-root",
    str(CODE_ROOT),
    "--pacs-source-root",
    str(SOURCE_ROOT),
    "--protocol",
    str(SOURCE_PROTOCOL),
    "--primary-training-root",
    str(PRIMARY_TRAINING),
    "--variant-root",
    str(VARIANT_ROOT),
    "--primary-diagnostics-root",
    str(PRIMARY_DIAGNOSTICS),
    "--output",
    str(DIAGNOSTICS_RESULT),
]

print("All Block 14 pre-diagnostic gates passed.")
print("Training commit preserved:", TRAINING_COMMIT)
print("Diagnostic commit:", diagnostic_commit)
print("Target-free tests passed: 29/29")
print("Previously completed and authenticated: ERM, original lambda 1, SAM")
print("New diagnostics: original lambda 0.1 and 10; stabilized lambda 1, 0.1, and 10")
print("Unified final coverage: all eight completed checkpoints")
print("Probe and sharpness subsets: exact Block 09 designs reused")
print("Sketch images accessible to diagnostics: 0\n")
subprocess.run(command, cwd=CODE_ROOT, check=True)

result = load_json(DIAGNOSTICS_RESULT, "additional source-diagnostic result")
for name, value in {
    "status": "TASK3_ADDITIONAL_SOURCE_DIAGNOSTICS_PASS",
    "scope": "missing_primary_strengths_and_bandwidth_floor_variants",
    "variant_id": VARIANT_ID,
    "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
    "contains_research_variant_models": True,
    "primary_protocol_replacement": False,
    "source_only_phase": True,
    "diagnostics_completed": True,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "primary_training_commit": PRIMARY_TRAINING_COMMIT,
    "primary_training_code_tree_sha256": PRIMARY_TRAINING_CODE_SHA256,
    "variant_training_commit": TRAINING_COMMIT,
    "variant_training_code_tree_sha256": TRAINING_CODE_SHA256,
    "diagnostic_code_tree_sha256": EXPECTED_DIAGNOSTIC_CODE_SHA256,
}.items():
    if result.get(name) != value:
        raise RuntimeError(f"Additional diagnostic result field differs: {name}")
new_model_names = set(PRIMARY_RUN_COMPLETIONS) | set(RUN_COMPLETIONS)
if set(result.get("models", {})) != new_model_names:
    raise RuntimeError("Additional diagnostics did not report all five pending models")

primary_result = load_json(
    PRIMARY_DIAGNOSTIC_RESULT,
    "Block 09 primary source-diagnostic result",
)
for name, value in {
    "status": "TASK3_SOURCE_DIAGNOSTICS_PASS",
    "source_only_phase": True,
    "diagnostics_completed": True,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
}.items():
    if primary_result.get(name) != value:
        raise RuntimeError(f"Block 09 source-diagnostic result differs: {name}")
if set(primary_result.get("models", {})) != {"erm", "dan_dg_1", "sam"}:
    raise RuntimeError("Block 09 does not contain the three official main models")

all_models = {
    "erm": primary_result["models"]["erm"],
    "dan_dg_0p1": result["models"]["dan_dg_0p1"],
    "dan_dg_1": primary_result["models"]["dan_dg_1"],
    "dan_dg_10": result["models"]["dan_dg_10"],
    "sam": primary_result["models"]["sam"],
    "dan_dg_floor_0p1": result["models"]["dan_dg_floor_0p1"],
    "dan_dg_floor_1": result["models"]["dan_dg_floor_1"],
    "dan_dg_floor_10": result["models"]["dan_dg_floor_10"],
}
unified = {
    "status": "TASK3_ALL_SOURCE_DIAGNOSTICS_COVERAGE_PASS",
    "source_only_phase": True,
    "diagnostics_completed": True,
    "complete_model_count": 8,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "primary_block_09_result": {
        "path": str(PRIMARY_DIAGNOSTIC_RESULT),
        "sha256": sha256_file(PRIMARY_DIAGNOSTIC_RESULT),
        "reused_models": ["erm", "dan_dg_1", "sam"],
    },
    "additional_block_14_result": {
        "path": str(DIAGNOSTICS_RESULT),
        "sha256": sha256_file(DIAGNOSTICS_RESULT),
        "newly_evaluated_models": list(result["models"]),
    },
    "reused_primary_probe_design": result["reused_primary_probe_design"],
    "reused_primary_sharpness_design": result[
        "reused_primary_sharpness_design"
    ],
    "models": all_models,
}
atomic_write_json(unified, ALL_MODELS_RESULT)

summary = {}
for model_name, model in all_models.items():
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
    "status": "TASK3_BLOCK_14_ALL_SOURCE_DIAGNOSTICS_COVERAGE_PASS",
    "variant_id": VARIANT_ID,
    "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
    "research_variant": True,
    "primary_protocol_replacement": False,
    "source_only_phase": True,
    "diagnostics_completed": True,
    "experiment_lock_created": False,
    "final_evaluation_started": False,
    "sketch_images_accessed": 0,
    "primary_training_commit": PRIMARY_TRAINING_COMMIT,
    "primary_training_code_tree_sha256": PRIMARY_TRAINING_CODE_SHA256,
    "variant_training_commit": TRAINING_COMMIT,
    "variant_training_code_tree_sha256": TRAINING_CODE_SHA256,
    "diagnostic_commit": diagnostic_commit,
    "diagnostic_code_tree_sha256": actual_code_sha256,
    "target_free_tests": {"count": 29, "passed": 29, "failed": 0},
    "authorization": {
        "path": str(AUTHORIZATION),
        "sha256": sha256_file(AUTHORIZATION),
    },
    "additional_result": {
        "path": str(DIAGNOSTICS_RESULT),
        "sha256": sha256_file(DIAGNOSTICS_RESULT),
    },
    "all_models_result": {
        "path": str(ALL_MODELS_RESULT),
        "sha256": sha256_file(ALL_MODELS_RESULT),
        "model_count": 8,
    },
    "reused_primary_probe_design": result["reused_primary_probe_design"],
    "reused_primary_sharpness_design": result[
        "reused_primary_sharpness_design"
    ],
    "summary": summary,
}
atomic_write_json(completion, COMPLETION)

print("\nAudited all-model source-diagnostic summary:\n")
print(json.dumps(completion, indent=2))
print("\nSaved at:", COMPLETION)
print("Sketch images accessed: 0")
print("Experiment lock created: False")
print("Final evaluation started: False")
print("Send the complete output for review before any final-lock decision.")
