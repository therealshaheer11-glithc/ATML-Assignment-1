"""Freeze or verify the manually reviewed 200-image AdaIN evaluation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image


REJECTION_REASONS = (
    "content object is no longer visually recognizable",
    "object is severely obscured or structurally corrupted",
    "output contains blank regions or major rendering artifacts",
    "no perceptible style or texture transfer occurred",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields:
        raise RuntimeError(f"CSV has no header: {path}")
    return fields, rows


def canonical_csv(rows: list[dict[str, str]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def require_equal(path: Path, expected: bytes) -> None:
    if not path.is_file() or path.read_bytes() != expected:
        raise RuntimeError(f"Frozen output missing or different: {path}")


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require_equal(path, data)
    else:
        with path.open("xb") as file:
            file.write(data)


def require_no_evaluation_outputs(artifacts_root: Path) -> None:
    for directory in (
        artifacts_root / "metrics/cue_conflict",
        artifacts_root / "predictions/cue_conflict",
    ):
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                "Cue-conflict model outputs already exist; selection may only be "
                "frozen before model evaluation. Use --mode verify for a completed run."
            )


def validate_candidate_image(artifacts_root: Path, row: dict[str, str]) -> None:
    image_path = (
        artifacts_root / "cue_conflict/candidates"
        / f"{row['candidate_id']}.png"
    )
    if not image_path.is_file() or digest(image_path.read_bytes()) != row["output_sha256"]:
        raise RuntimeError(f"Candidate image changed: {row['candidate_id']}")
    with Image.open(image_path) as image:
        if image.mode != "RGB" or image.size != (224, 224):
            raise RuntimeError(f"Invalid candidate image: {row['candidate_id']}")


def build_outputs(
    artifacts_root: Path,
) -> tuple[bytes, bytes, bytes]:
    manifest_dir = artifacts_root / "cue_conflict/manifests"
    review_path = manifest_dir / "review_manifest.csv"
    fields, rows = read_csv(review_path)
    _, initial = read_csv(manifest_dir / "review_snapshot_round_01.csv")
    _, before_round_2 = read_csv(manifest_dir / "review_snapshot_round_02.csv")
    if len(rows) != 265 or rows[:200] != initial or rows[:256] != before_round_2:
        raise RuntimeError("Review history differs from the recorded two-round workflow.")
    if len({row["candidate_id"] for row in rows}) != 265:
        raise RuntimeError("Review manifest contains duplicate candidate IDs.")

    plans = [
        json.loads(
            (manifest_dir / f"replacement_plan_round_{number:02d}.json")
            .read_text(encoding="utf-8")
        )
        for number in (1, 2)
    ]
    if [row["candidate_id"] for row in rows[200:256]] != [
        item["candidate_id"] for item in plans[0]["planned_candidates"]
    ]:
        raise RuntimeError("Round-1 rows differ from the replacement plan.")
    if [row["candidate_id"] for row in rows[256:]] != [
        item["candidate_id"] for item in plans[1]["planned_candidates"]
    ]:
        raise RuntimeError("Round-2 rows differ from the replacement plan.")

    for number in (1, 2):
        metadata = json.loads(
            (manifest_dir / f"replacement_generation_round_{number:02d}.json")
            .read_text(encoding="utf-8")
        )
        if (
            metadata["generation_device"] != "cpu"
            or metadata["model_predictions_run_before_manual_review"] is not False
        ):
            raise RuntimeError(f"Invalid replacement metadata for round {number}.")

    statuses = Counter(row["review_status"] for row in rows)
    if statuses != {"accepted": 200, "rejected": 65}:
        raise RuntimeError(f"Unexpected review totals: {dict(statuses)}")
    for row in rows:
        if (
            not row["reviewed_by"]
            or row["review_completed_before_model_evaluation"] != "True"
            or row["model_predictions_used_for_generation"] != "False"
        ):
            raise RuntimeError(f"Incomplete review provenance: {row['candidate_id']}")
        if row["review_status"] == "accepted" and row["rejection_reason"]:
            raise RuntimeError(f"Accepted row has a rejection reason: {row['candidate_id']}")
        if (
            row["review_status"] == "rejected"
            and row["rejection_reason"] not in REJECTION_REASONS
        ):
            raise RuntimeError(f"Invalid rejection reason: {row['candidate_id']}")

        validate_candidate_image(artifacts_root, row)

    accepted = [row for row in rows if row["review_status"] == "accepted"]
    directions = Counter(row["direction"] for row in accepted)
    pairs = Counter(row["pair_number"] for row in accepted)
    if len(directions) != 10 or set(directions.values()) != {20}:
        raise RuntimeError("Accepted set is not balanced at 20 per direction.")
    if len(pairs) != 5 or set(pairs.values()) != {40}:
        raise RuntimeError("Accepted set is not balanced at 40 per pair.")

    accepted.sort(
        key=lambda row: (
            int(row["pair_number"]),
            row["direction_code"],
            int(row["schedule_rank"]),
        )
    )
    review_data = review_path.read_bytes()
    accepted_data = canonical_csv(accepted, fields)
    record = {
        "candidate_count": 265,
        "accepted_count": 200,
        "rejected_count": 65,
        "accepted_per_direction": dict(sorted(directions.items())),
        "accepted_per_pair": dict(sorted(pairs.items())),
        "final_review_manifest_sha256": digest(review_data),
        "accepted_conflicts_sha256": digest(accepted_data),
        "replacement_generation_device_rounds_1_and_2": "cpu",
        "model_predictions_used_for_selection": False,
        "model_evaluation_started_at_freeze": False,
    }
    record_data = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    return review_data, accepted_data, record_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "verify"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    artifacts = args.artifacts_root.resolve()
    outputs = dict(zip(
        ("final_review_manifest.csv", "accepted_conflicts.csv", "selection_record.json"),
        build_outputs(artifacts),
    ))
    destinations = (
        artifacts / "cue_conflict/manifests",
        repo / "task1/results/cue_conflict",
    )
    if args.mode == "freeze":
        require_no_evaluation_outputs(artifacts)
        for directory in destinations:
            for name, data in outputs.items():
                write_once(directory / name, data)
        print("PASS: selection frozen before model evaluation.")
    else:
        for directory in destinations:
            for name, data in outputs.items():
                require_equal(directory / name, data)
        print("PASS: frozen selection matches the complete review history.")
    print("PASS: 200 accepted, 65 rejected, 20 accepted per direction.")
    print("PASS: all candidate image hashes and review-provenance fields verified.")


if __name__ == "__main__":
    main()
