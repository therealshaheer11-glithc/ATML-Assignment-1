# TASK 3 — BLOCK 05 COMPLETION AUDIT
# Audits DAN-DG lambda=1 artifacts and writes its completion record.
# No dataset images are opened.

from pathlib import Path
import csv
import hashlib
import json
import math
import os
import subprocess

import torch


RUN_ID = "dan_dg_1"

EXPECTED_COMMIT = "19208b4c62acb980fb3246f30e062784b90d8dfc"
EXPECTED_CODE_SHA256 = (
    "4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44"
)
EXPECTED_PROTOCOL_SHA256 = (
    "626d8517b44ad50c0219adf49e827de6538561386791bed29a9153a589cd6abc"
)
EXPECTED_SOURCE_SNAPSHOT_SHA256 = (
    "8ded350769ee15739f8420e755e50ff4377068a4f54ab1c0ba39d5b125e658d2"
)
EXPECTED_INITIALIZATION_SHA256 = (
    "4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3"
)
EXPECTED_SHARED_MMD_SHA256 = (
    "cfe0b1d9c22d7f492ea5e8f76732fbabf21c86cb53f24759af65fda09f9bfbcc"
)
EXPECTED_CODE_PREFLIGHT_SHA256 = (
    "40ef37d7ae08ece5526e5588e446e5301158cc4bd444e2213cedfc0a9bf73eee"
)

CODE_ROOT = Path("/content/atml_pa1_task3_source")
TASK3_ROOT = Path(
    "/content/drive/MyDrive/ATML-PA1/"
    "task3_domain_generalization_20260924"
)
RUN_DIRECTORY = TASK3_ROOT / "training" / RUN_ID
PROVENANCE_ROOT = TASK3_ROOT / "provenance"

RUN_MANIFEST = RUN_DIRECTORY / "run.json"
HISTORY = RUN_DIRECTORY / "history.csv"
BEST_VALIDATION = RUN_DIRECTORY / "best_source_validation.json"
BEST_CHECKPOINT = RUN_DIRECTORY / "best.pt"
LAST_CHECKPOINT = RUN_DIRECTORY / "last.pt"
COMPLETION_RECORD = (
    PROVENANCE_ROOT
    / "dan_dg_1_training_completion.json"
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


def atomic_write_json(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    os.replace(temporary, path)


# Confirm all expected training artifacts exist.
required_files = (
    RUN_MANIFEST,
    HISTORY,
    BEST_VALIDATION,
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
)

missing = [
    str(path)
    for path in required_files
    if not path.is_file()
]

if missing:
    raise FileNotFoundError(
        f"Missing DAN-DG artifacts: {missing}"
    )


# Inspect the final run manifest.
manifest = json.loads(RUN_MANIFEST.read_text())

required_manifest_fields = {
    "status": "TASK3_RUN_COMPLETE",
    "run_id": RUN_ID,
    "steps_per_source_epoch": 235,
    "source_only_phase": True,
    "sketch_images_accessed": 0,
}

for field, expected in required_manifest_fields.items():
    actual = manifest.get(field)
    if actual != expected:
        raise RuntimeError(
            f"Manifest field {field!r} differs: "
            f"{actual!r} != {expected!r}"
        )

identity = manifest.get("identity", {})
config = identity.get("config", {})

required_config = {
    "run_id": RUN_ID,
    "method": "dan_dg",
    "mmd_lambda": 1.0,
    "sam_rho": None,
    "epochs": 30,
    "patience": 5,
    "source_batch_per_domain": 8,
    "steps_per_source_epoch": 235,
    "gradient_clipping": 20.0,
    "mmd_feature_normalization":
        "l2_per_sample_mmd_input_only",
    "adversarial_feature_normalization":
        "not_applicable_to_task3",
}

for field, expected in required_config.items():
    actual = config.get(field)
    if actual != expected:
        raise RuntimeError(
            f"Configuration field {field!r} differs: "
            f"{actual!r} != {expected!r}"
        )

required_identities = {
    "code_sha256": EXPECTED_CODE_SHA256,
    "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    "source_snapshot_sha256":
        EXPECTED_SOURCE_SNAPSHOT_SHA256,
    "initialization_sha256":
        EXPECTED_INITIALIZATION_SHA256,
    "shared_mmd_sha256":
        EXPECTED_SHARED_MMD_SHA256,
    "code_preflight_sha256":
        EXPECTED_CODE_PREFLIGHT_SHA256,
    "source_only_phase": True,
    "sketch_images_accessed": 0,
}

for field, expected in required_identities.items():
    actual = identity.get(field)
    if actual != expected:
        raise RuntimeError(
            f"Run identity {field!r} differs: "
            f"{actual!r} != {expected!r}"
        )

current_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=CODE_ROOT,
    text=True,
).strip()

if current_commit != EXPECTED_COMMIT:
    raise RuntimeError(
        "Repository commit changed during training:\n"
        f"Expected: {EXPECTED_COMMIT}\n"
        f"Actual:   {current_commit}"
    )


# Inspect every completed epoch.
with HISTORY.open(newline="") as handle:
    history = list(csv.DictReader(handle))

epochs_completed = manifest.get("epochs_completed")
best_epoch = manifest.get("best_epoch")
best_f1 = float(
    manifest.get("best_mean_source_macro_f1")
)

if len(history) != epochs_completed:
    raise RuntimeError(
        "History length differs from epochs_completed"
    )

if not 1 <= epochs_completed <= 30:
    raise RuntimeError(
        f"Invalid completed epoch count: {epochs_completed}"
    )

if [int(row["epoch"]) for row in history] != list(
    range(1, epochs_completed + 1)
):
    raise RuntimeError(
        "History epochs are missing, duplicated, or unordered"
    )

if not 1 <= best_epoch <= epochs_completed:
    raise RuntimeError(
        f"Invalid selected epoch: {best_epoch}"
    )

if not math.isfinite(best_f1):
    raise RuntimeError(
        "Best source-validation macro-F1 is non-finite"
    )

finite_columns = (
    "classification_loss",
    "mmd_loss",
    "mmd_photo__art_painting",
    "mmd_photo__cartoon",
    "mmd_art_painting__cartoon",
    "mmd_median_photo__art_painting",
    "mmd_median_photo__cartoon",
    "mmd_median_art_painting__cartoon",
    "gradient_norm",
    "gradient_norm_after_clipping",
    "gradient_clipped_fraction",
    "mean_source_accuracy",
    "mean_source_macro_f1",
    "worst_source_accuracy",
    "worst_source_macro_f1",
)

for row in history:
    epoch = int(row["epoch"])

    for column in finite_columns:
        value = float(row[column])
        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite {column} at epoch {epoch}"
            )

    for column in (
        "mmd_median_photo__art_painting",
        "mmd_median_photo__cartoon",
        "mmd_median_art_painting__cartoon",
    ):
        if float(row[column]) <= 0:
            raise RuntimeError(
                f"Non-positive {column} at epoch {epoch}"
            )

    if float(
        row["gradient_norm_after_clipping"]
    ) > 20.001:
        raise RuntimeError(
            f"Gradient clipping limit exceeded at epoch {epoch}"
        )

    clipped_fraction = float(
        row["gradient_clipped_fraction"]
    )
    if not 0.0 <= clipped_fraction <= 1.0:
        raise RuntimeError(
            f"Invalid clipping fraction at epoch {epoch}"
        )

    for column in (
        "mean_source_accuracy",
        "mean_source_macro_f1",
        "worst_source_accuracy",
        "worst_source_macro_f1",
    ):
        if not 0.0 <= float(row[column]) <= 1.0:
            raise RuntimeError(
                f"Invalid metric {column} at epoch {epoch}"
            )


# Verify strict-improvement checkpoint selection.
history_f1 = [
    float(row["mean_source_macro_f1"])
    for row in history
]
calculated_best_f1 = max(history_f1)
calculated_best_epoch = history_f1.index(
    calculated_best_f1
) + 1

if calculated_best_epoch != best_epoch:
    raise RuntimeError(
        "Recorded best epoch violates earliest strict tie policy"
    )

if abs(calculated_best_f1 - best_f1) > 1e-12:
    raise RuntimeError(
        "Recorded best source macro-F1 differs from history"
    )

if epochs_completed < 30:
    stale_epochs = epochs_completed - best_epoch
    if stale_epochs < 5:
        raise RuntimeError(
            "Run stopped early before five stale epochs"
        )


# Verify selected-validation and checkpoint artifacts.
best_validation = json.loads(
    BEST_VALIDATION.read_text()
)

if best_validation.get("epoch") != best_epoch:
    raise RuntimeError(
        "Best-validation JSON has the wrong epoch"
    )

if abs(
    float(best_validation["mean_source_macro_f1"])
    - best_f1
) > 1e-12:
    raise RuntimeError(
        "Best-validation JSON disagrees with run.json"
    )

best_checkpoint_sha256 = sha256_file(
    BEST_CHECKPOINT
)

if (
    best_checkpoint_sha256
    != manifest.get("best_checkpoint_sha256")
):
    raise RuntimeError(
        "Selected-checkpoint hash differs from run.json"
    )

checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

if checkpoint.get("epoch") != best_epoch:
    raise RuntimeError(
        "Selected checkpoint has the wrong epoch"
    )

if checkpoint.get("identity") != identity:
    raise RuntimeError(
        "Selected checkpoint has a different run identity"
    )

if checkpoint.get("source_only_phase") is not True:
    raise RuntimeError(
        "Selected checkpoint is not source-only"
    )

if checkpoint.get("sketch_images_accessed") != 0:
    raise RuntimeError(
        "Selected checkpoint reports Sketch access"
    )

checkpoint_f1 = float(
    checkpoint["source_validation"][
        "mean_source_macro_f1"
    ]
)

if abs(checkpoint_f1 - best_f1) > 1e-12:
    raise RuntimeError(
        "Selected checkpoint metric disagrees with manifest"
    )


# Persist the audited completion record.
completion = {
    "status": "TASK3_BLOCK_05_DAN_DG_1_PASS",
    "protocol_version":
        "task3-approved-2026-09-24-v1",
    "run_id": RUN_ID,
    "method": "dan_dg",
    "mmd_lambda": 1.0,
    "source_only_phase": True,
    "training_completed": True,
    "sketch_images_accessed": 0,
    "repository_commit": current_commit,
    "code_tree_sha256": identity["code_sha256"],
    "epochs_completed": epochs_completed,
    "best_epoch": best_epoch,
    "best_mean_source_macro_f1": best_f1,
    "best_source_validation":
        checkpoint["source_validation"],
    "best_checkpoint": {
        "path": str(BEST_CHECKPOINT),
        "sha256": best_checkpoint_sha256,
    },
    "history": {
        "path": str(HISTORY),
        "sha256": sha256_file(HISTORY),
        "rows": len(history),
    },
    "run_manifest": {
        "path": str(RUN_MANIFEST),
        "sha256": sha256_file(RUN_MANIFEST),
    },
}

if COMPLETION_RECORD.exists():
    existing = json.loads(
        COMPLETION_RECORD.read_text()
    )
    if existing != completion:
        raise RuntimeError(
            "A different DAN-DG completion record exists"
        )
else:
    atomic_write_json(
        completion,
        COMPLETION_RECORD,
    )


print("DAN-DG lambda=1 epoch summary:\n")

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
print(
    f"\nSaved at: {COMPLETION_RECORD}"
)
print("Sketch images accessed: 0")
print(
    "Do not start another run until this output is reviewed."
)
