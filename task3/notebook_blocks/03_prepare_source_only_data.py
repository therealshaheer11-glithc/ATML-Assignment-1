"""Task 3 notebook block 03: prepare a source-only PACS workspace.

Prerequisites:
  * Blocks 01 and 02 passed in this runtime.
  * Google Drive is already mounted.

This block clones the public repository only to obtain the approved Task 2 split
manifest, verifies that manifest and the PACS archive, creates a target-free protocol,
and extracts exactly the approved Photo/Art/Cartoon records. It never opens or extracts
a Sketch image member.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image


PROTOCOL_VERSION = "task3-approved-2026-09-24-v1"
SEED = 6304
SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
FORBIDDEN_TARGET_DOMAIN = "sketch"
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_TASK2_PROTOCOL_SHA256 = (
    "e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74"
)
EXPECTED_DATASET_FILE_LIST_SHA256 = (
    "559ac63b8df8e07330b97112e5ec4c414b3957585b28d21b4cfecd2181f538e0"
)
EXPECTED_PACS_ARCHIVE_SHA256 = (
    "0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102"
)
EXPECTED_ARCHIVE_MEMBERS = 10044
EXPECTED_SOURCE_COUNTS = {
    "photo": {"train": 1336, "validation": 334},
    "art_painting": {"train": 1638, "validation": 410},
    "cartoon": {"train": 1875, "validation": 469},
}
EXPECTED_SOURCE_IMAGES = sum(
    split_count
    for domain_counts in EXPECTED_SOURCE_COUNTS.values()
    for split_count in domain_counts.values()
)

REPOSITORY_URL = "https://github.com/therealshaheer11-glithc/ATML-Assignment-1.git"
CODE_ROOT = Path("/content/atml_pa1_task3_source")

ATML_DRIVE_ROOT = Path("/content/drive/MyDrive/ATML-PA1")
TASK3_DRIVE_ROOT = ATML_DRIVE_ROOT / "task3_domain_generalization_20260924"
TASK3_PROVENANCE = TASK3_DRIVE_ROOT / "provenance"
TASK3_PROTOCOL_DIR = TASK3_DRIVE_ROOT / "source_protocol"

BLOCK02_RECORD = TASK3_PROVENANCE / "task2_artifact_verification.json"
DRIVE_ARCHIVE = ATML_DRIVE_ROOT / "datasets" / "PACS_dassl.zip"

TASK2_PROTOCOL_RELATIVE = Path("shared/splits/pacs_sketch_seed6304.json")
SOURCE_PROTOCOL_PATH = TASK3_PROTOCOL_DIR / "pacs_sources_seed6304.json"
SOURCE_PREPARATION_RECORD = TASK3_PROVENANCE / "source_data_preparation.json"

SOURCE_ROOT = Path("/content/task3_pacs_sources_v1")
PARTIAL_SOURCE_ROOT = Path("/content/task3_pacs_sources_v1.partial")
LOCAL_SENTINEL = SOURCE_ROOT / "SOURCE_SNAPSHOT.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write_bytes(data: bytes, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    atomic_write_bytes(canonical_json_bytes(payload), path)


def approved_relative_path(member_name: str, expected_paths: set[str]) -> str | None:
    """Map one archive member to an approved source-relative path, if applicable."""
    path = PurePosixPath(member_name)
    parts = path.parts

    if not parts or path.is_absolute() or ".." in parts:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")

    lowered = tuple(part.lower() for part in parts)
    source_positions = [
        index for index, part in enumerate(lowered) if part in SOURCE_DOMAINS
    ]

    if not source_positions:
        # Metadata and non-source members are deliberately not opened.
        return None

    if len(source_positions) != 1:
        raise RuntimeError(f"Ambiguous source-domain path: {member_name}")

    source_index = source_positions[0]
    relative = PurePosixPath(*parts[source_index:]).as_posix()

    if FORBIDDEN_TARGET_DOMAIN in {part.lower() for part in parts}:
        raise RuntimeError(f"A source candidate unexpectedly contains Sketch: {member_name}")

    if relative not in expected_paths:
        return None

    return relative


def validate_task2_protocol(protocol: dict) -> set[str]:
    expected_header = {
        "dataset": "PACS",
        "seed": SEED,
        "sources": list(SOURCE_DOMAINS),
        "target": FORBIDDEN_TARGET_DOMAIN,
        "classes": list(CLASSES),
        "file_list_sha256": EXPECTED_DATASET_FILE_LIST_SHA256,
    }

    for name, expected in expected_header.items():
        if protocol.get(name) != expected:
            raise RuntimeError(
                f"Task 2 protocol field {name!r} differs: "
                f"expected {expected!r}, got {protocol.get(name)!r}"
            )

    expected_paths: set[str] = set()

    for domain in SOURCE_DOMAINS:
        for split_name in ("train", "validation"):
            rows = protocol.get("source_splits", {}).get(domain, {}).get(split_name, [])
            expected_count = EXPECTED_SOURCE_COUNTS[domain][split_name]

            if len(rows) != expected_count:
                raise RuntimeError(
                    f"Unexpected {domain}/{split_name} count: "
                    f"{len(rows)} != {expected_count}"
                )

            class_counts = [0] * len(CLASSES)

            for row in rows:
                if set(row) != {"path", "class_id"}:
                    raise RuntimeError(
                        f"Invalid source record fields in {domain}/{split_name}: {row}"
                    )

                relative = PurePosixPath(str(row["path"]))
                parts = relative.parts
                class_id = row["class_id"]

                if not parts or parts[0] != domain:
                    raise RuntimeError(f"Wrong-domain source record: {row}")
                if FORBIDDEN_TARGET_DOMAIN in {part.lower() for part in parts}:
                    raise RuntimeError(f"Sketch record found in a source split: {row}")
                if not isinstance(class_id, int) or not 0 <= class_id < len(CLASSES):
                    raise RuntimeError(f"Invalid source class ID: {row}")
                if row["path"] in expected_paths:
                    raise RuntimeError(f"Duplicate source membership: {row['path']}")

                expected_paths.add(row["path"])
                class_counts[class_id] += 1

            if any(count == 0 for count in class_counts):
                raise RuntimeError(f"A class is missing from {domain}/{split_name}")

    if len(expected_paths) != EXPECTED_SOURCE_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_IMAGES} unique source records, "
            f"found {len(expected_paths)}"
        )

    return expected_paths


def verify_local_source_snapshot(
    root: Path,
    expected_paths: set[str],
) -> tuple[str, dict[str, int]]:
    actual_paths: set[str] = set()
    counts = {domain: 0 for domain in SOURCE_DOMAINS}
    digest = hashlib.sha256()
    unreadable: list[str] = []

    for domain in SOURCE_DOMAINS:
        domain_root = root / domain
        if not domain_root.is_dir():
            raise RuntimeError(f"Missing extracted source domain: {domain_root}")

        files = sorted(
            path
            for path in domain_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for path in files:
            relative = path.relative_to(root).as_posix()
            actual_paths.add(relative)
            counts[domain] += 1
            digest.update(f"{relative}\t{path.stat().st_size}\n".encode("utf-8"))

            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                unreadable.append(relative)

    if unreadable:
        raise RuntimeError(f"Unreadable source images: {unreadable[:5]}")

    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing or unexpected:
        raise RuntimeError(
            "Extracted source set differs from the approved manifest. "
            f"Missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    if len(actual_paths) != EXPECTED_SOURCE_IMAGES:
        raise RuntimeError(
            f"Unexpected extracted source count: {len(actual_paths)}"
        )

    return digest.hexdigest(), counts


if not BLOCK02_RECORD.is_file():
    raise FileNotFoundError(
        "Block 02 artifact-verification record is missing. Run Block 02 first."
    )

block02 = json.loads(BLOCK02_RECORD.read_text())
if block02.get("status") != "TASK3_BLOCK_02_PASS":
    raise RuntimeError("Block 02 did not record a passing status.")
if block02.get("source_only_phase") is not True:
    raise RuntimeError("Block 02 did not preserve the source-only phase.")
if block02.get("sketch_images_accessed") != 0:
    raise RuntimeError("Block 02 reports unexpected Sketch access.")

if not DRIVE_ARCHIVE.is_file():
    raise FileNotFoundError(f"Verified PACS archive is missing: {DRIVE_ARCHIVE}")

archive_sha256 = sha256_file(DRIVE_ARCHIVE)
if archive_sha256 != EXPECTED_PACS_ARCHIVE_SHA256:
    raise RuntimeError(
        "PACS archive hash mismatch.\n"
        f"Expected: {EXPECTED_PACS_ARCHIVE_SHA256}\n"
        f"Actual:   {archive_sha256}"
    )

if CODE_ROOT.exists():
    raise FileExistsError(
        f"Fresh clone path already exists: {CODE_ROOT}. "
        "Do not overwrite it; report this state before continuing."
    )

subprocess.run(
    ["git", "clone", "--depth", "1", REPOSITORY_URL, str(CODE_ROOT)],
    check=True,
)

repository_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=CODE_ROOT,
    text=True,
).strip()

task2_protocol_path = CODE_ROOT / TASK2_PROTOCOL_RELATIVE
if not task2_protocol_path.is_file():
    raise FileNotFoundError(
        f"Approved Task 2 split manifest is missing: {task2_protocol_path}"
    )

task2_protocol_sha256 = sha256_file(task2_protocol_path)
if task2_protocol_sha256 != EXPECTED_TASK2_PROTOCOL_SHA256:
    raise RuntimeError(
        "Task 2 split-manifest hash mismatch.\n"
        f"Expected: {EXPECTED_TASK2_PROTOCOL_SHA256}\n"
        f"Actual:   {task2_protocol_sha256}\n"
        f"Repository commit: {repository_commit}"
    )

task2_protocol = json.loads(task2_protocol_path.read_text())
expected_paths = validate_task2_protocol(task2_protocol)

source_protocol = {
    "dataset": "PACS",
    "seed": SEED,
    "sources": list(SOURCE_DOMAINS),
    "classes": list(CLASSES),
    "parent_task2_protocol_sha256": task2_protocol_sha256,
    "parent_dataset_file_list_sha256": EXPECTED_DATASET_FILE_LIST_SHA256,
    "pacs_archive_sha256": archive_sha256,
    "target_domain_embargoed": FORBIDDEN_TARGET_DOMAIN,
    "target_records_included": False,
    "source_splits": task2_protocol["source_splits"],
}

source_protocol_bytes = canonical_json_bytes(source_protocol)
source_protocol_sha256 = hashlib.sha256(source_protocol_bytes).hexdigest()

TASK3_PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
if SOURCE_PROTOCOL_PATH.exists():
    if SOURCE_PROTOCOL_PATH.read_bytes() != source_protocol_bytes:
        raise RuntimeError(
            "A different source-only protocol already exists in Task 3 Drive storage."
        )
else:
    atomic_write_bytes(source_protocol_bytes, SOURCE_PROTOCOL_PATH)

if SOURCE_ROOT.exists():
    if not LOCAL_SENTINEL.is_file():
        raise RuntimeError(
            f"Source workspace exists without a verified sentinel: {SOURCE_ROOT}"
        )

    local_record = json.loads(LOCAL_SENTINEL.read_text())
    if local_record.get("source_protocol_sha256") != source_protocol_sha256:
        raise RuntimeError("Existing source workspace uses a different protocol.")

    source_snapshot_sha256, extracted_counts = verify_local_source_snapshot(
        SOURCE_ROOT,
        expected_paths,
    )
    if source_snapshot_sha256 != local_record.get("source_snapshot_sha256"):
        raise RuntimeError("Existing source workspace snapshot hash changed.")
    reused_existing_workspace = True
else:
    if PARTIAL_SOURCE_ROOT.exists():
        raise RuntimeError(
            f"Partial source workspace already exists: {PARTIAL_SOURCE_ROOT}. "
            "Restart the runtime before retrying; do not merge partial extraction."
        )

    PARTIAL_SOURCE_ROOT.mkdir(parents=True)
    extracted_paths: set[str] = set()

    with zipfile.ZipFile(DRIVE_ARCHIVE) as archive:
        members = archive.infolist()
        if len(members) != EXPECTED_ARCHIVE_MEMBERS:
            raise RuntimeError(
                f"Unexpected PACS archive member count: {len(members)}"
            )

        for member in members:
            if member.is_dir():
                continue

            relative = approved_relative_path(member.filename, expected_paths)
            if relative is None:
                continue

            if relative in extracted_paths:
                raise RuntimeError(f"Duplicate approved archive member: {relative}")

            destination = (PARTIAL_SOURCE_ROOT / relative).resolve()
            partial_base = PARTIAL_SOURCE_ROOT.resolve()
            if partial_base not in destination.parents:
                raise RuntimeError(f"Unsafe extraction destination: {destination}")

            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

            extracted_paths.add(relative)

    missing_after_extraction = sorted(expected_paths - extracted_paths)
    if missing_after_extraction:
        raise RuntimeError(
            "Approved source members were not found in the archive: "
            f"{missing_after_extraction[:5]}"
        )
    if len(extracted_paths) != EXPECTED_SOURCE_IMAGES:
        raise RuntimeError(
            f"Extracted {len(extracted_paths)} source images; "
            f"expected {EXPECTED_SOURCE_IMAGES}"
        )

    source_snapshot_sha256, extracted_counts = verify_local_source_snapshot(
        PARTIAL_SOURCE_ROOT,
        expected_paths,
    )

    local_record = {
        "status": "VERIFIED_SOURCE_ONLY_PACS",
        "protocol_version": PROTOCOL_VERSION,
        "source_protocol_sha256": source_protocol_sha256,
        "parent_task2_protocol_sha256": task2_protocol_sha256,
        "pacs_archive_sha256": archive_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_image_count": len(extracted_paths),
        "per_domain_image_counts": extracted_counts,
        "sketch_members_opened_or_extracted": 0,
    }
    atomic_write_json(local_record, PARTIAL_SOURCE_ROOT / "SOURCE_SNAPSHOT.json")
    os.replace(PARTIAL_SOURCE_ROOT, SOURCE_ROOT)
    reused_existing_workspace = False

source_preparation_record = {
    "status": "TASK3_BLOCK_03_PASS",
    "protocol_version": PROTOCOL_VERSION,
    "source_only_phase": True,
    "repository": {
        "url": REPOSITORY_URL,
        "commit": repository_commit,
        "local_path": str(CODE_ROOT),
    },
    "task2_protocol": {
        "path": str(task2_protocol_path),
        "sha256": task2_protocol_sha256,
    },
    "source_protocol": {
        "path": str(SOURCE_PROTOCOL_PATH),
        "sha256": source_protocol_sha256,
        "target_records_included": False,
    },
    "pacs_archive": {
        "path": str(DRIVE_ARCHIVE),
        "sha256": archive_sha256,
        "member_count": EXPECTED_ARCHIVE_MEMBERS,
    },
    "source_workspace": {
        "path": str(SOURCE_ROOT),
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_image_count": EXPECTED_SOURCE_IMAGES,
        "per_domain_image_counts": extracted_counts,
        "reused_existing_workspace": reused_existing_workspace,
    },
    "dataset_images_opened_for_verification": EXPECTED_SOURCE_IMAGES,
    "sketch_images_opened_or_extracted": 0,
}

if SOURCE_PREPARATION_RECORD.exists():
    existing_record = json.loads(SOURCE_PREPARATION_RECORD.read_text())
    if existing_record != source_preparation_record:
        raise RuntimeError(
            "A different source-data preparation record already exists in Drive."
        )
else:
    atomic_write_json(source_preparation_record, SOURCE_PREPARATION_RECORD)

print(json.dumps(source_preparation_record, indent=2))
print(f"\nSource-only PACS root: {SOURCE_ROOT}")
print(f"Source-only protocol: {SOURCE_PROTOCOL_PATH}")
print(f"Saved preparation record: {SOURCE_PREPARATION_RECORD}")
print(f"Approved source images verified: {EXPECTED_SOURCE_IMAGES}")
print("Sketch images opened or extracted: 0")
print("Next action: inspect this record before installing Task 3 implementation code.")

