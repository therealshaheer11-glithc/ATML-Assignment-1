"""Generate only preregistered AdaIN replacements after manual review.

The first 200 candidates and their review are never overwritten. Planning and
generation use no classifier predictions. Each new review row is saved to Drive
immediately, so an interrupted CPU run can resume without losing progress.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.datasets import STL10


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from task1.adain.generate_conflicts import (  # noqa: E402
    ALPHA,
    AVAILABLE_RANKS_PER_DIRECTION,
    GENERATION_BATCH_SIZE,
    IMAGE_SIZE,
    INITIAL_RANKS_PER_DIRECTION,
    PAIRS,
    REJECTION_REASONS,
    load_adain,
    read_csv,
    save_png,
    seed_everything,
    sha256_file,
)
from task1.transforms import common_rgb_tensor  # noqa: E402


TARGET_PER_DIRECTION = 20
EXPECTED_DIRECTIONS = 2 * len(PAIRS)
REVIEW_EXTRA_FIELDS = (
    "output_path",
    "output_sha256",
    "output_width",
    "output_height",
    "pixel_minimum",
    "pixel_maximum",
    "pixel_mean",
    "pixel_standard_deviation",
    "model_predictions_used_for_generation",
    "review_status",
    "rejection_reason",
    "reviewed_by",
    "review_completed_before_model_evaluation",
)


def canonical_csv_bytes(rows: list[dict[str, str]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def read_review(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise RuntimeError("Review manifest is empty or malformed.")
    if any(field not in fields for field in REVIEW_EXTRA_FIELDS):
        raise RuntimeError("Review manifest lacks required fields.")
    return fields, rows


def validate_review(
    review_rows: list[dict[str, str]],
    schedule_rows: list[dict[str, str]],
    candidate_dir: Path,
) -> dict[str, list[dict[str, str]]]:
    schedule = {row["candidate_id"]: row for row in schedule_rows}
    if len(schedule_rows) != 500 or len(schedule) != 500:
        raise RuntimeError("Expected the immutable 500-candidate schedule.")
    if len(review_rows) < 200 or len({r["candidate_id"] for r in review_rows}) != len(review_rows):
        raise RuntimeError("Review rows are missing or contain duplicate IDs.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for review in review_rows:
        candidate_id = review["candidate_id"]
        if candidate_id not in schedule:
            raise RuntimeError(f"Unscheduled candidate in review: {candidate_id}")
        scheduled = schedule[candidate_id]
        for field in scheduled:
            if review[field] != scheduled[field]:
                raise RuntimeError(f"Schedule/review mismatch: {candidate_id}, {field}")
        if review["review_status"] not in ("accepted", "rejected"):
            raise RuntimeError("Complete the current review before replacements.")
        if review["review_status"] == "rejected":
            if review["rejection_reason"] not in REJECTION_REASONS:
                raise RuntimeError(f"Invalid rejection reason for {candidate_id}.")
        elif review["rejection_reason"]:
            raise RuntimeError(f"Accepted candidate has rejection reason: {candidate_id}")
        if review["review_completed_before_model_evaluation"] != "True":
            raise RuntimeError(f"Review timing not recorded for {candidate_id}.")
        if not review["reviewed_by"]:
            raise RuntimeError(f"Reviewer not recorded for {candidate_id}.")

        image_path = candidate_dir / f"{candidate_id}.png"
        if Path(review["output_path"]) != image_path or not image_path.is_file():
            raise RuntimeError(f"Candidate image missing/misplaced: {candidate_id}")
        if sha256_file(image_path) != review["output_sha256"]:
            raise RuntimeError(f"Candidate image changed: {candidate_id}")
        grouped[review["direction"]].append(review)

    if len(grouped) != EXPECTED_DIRECTIONS:
        raise RuntimeError("Expected all ten class-pair directions.")
    for direction, rows in grouped.items():
        ranks = sorted(int(row["schedule_rank"]) for row in rows)
        if ranks != list(range(1, len(rows) + 1)):
            raise RuntimeError(f"Non-contiguous schedule ranks: {direction}")
        if not INITIAL_RANKS_PER_DIRECTION <= len(rows) <= AVAILABLE_RANKS_PER_DIRECTION:
            raise RuntimeError(f"Invalid candidate count for {direction}.")
        if sum(row["review_status"] == "accepted" for row in rows) > TARGET_PER_DIRECTION:
            raise RuntimeError(f"More than 20 accepted in {direction}.")
    return grouped


def load_state(
    repo_root: Path, artifacts_root: Path
) -> tuple[Path, list[dict[str, str]], Path, list[str], list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    schedule_path = repo_root / "task1/results/cue_conflict/candidate_schedule.csv"
    record_path = repo_root / "task1/results/cue_conflict/candidate_schedule.json"
    review_path = artifacts_root / "cue_conflict/manifests/review_manifest.csv"
    candidate_dir = artifacts_root / "cue_conflict/candidates"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["schedule_sha256"] != sha256_file(schedule_path):
        raise RuntimeError("The preregistered schedule has changed.")
    if record["candidate_images_present_at_schedule_creation"] != 0:
        raise RuntimeError("Schedule was not registered before generation.")
    schedule_rows = read_csv(schedule_path)
    fields, review_rows = read_review(review_path)
    grouped = validate_review(review_rows, schedule_rows, candidate_dir)
    return schedule_path, schedule_rows, review_path, fields, review_rows, grouped


def plan_path(artifacts_root: Path, round_number: int) -> Path:
    return (
        artifacts_root / "cue_conflict/manifests"
        / f"replacement_plan_round_{round_number:02d}.json"
    )


def create_plan(
    repo_root: Path,
    artifacts_root: Path,
    round_number: int,
    device: str,
) -> None:
    schedule_path, schedule_rows, review_path, fields, review_rows, grouped = load_state(
        repo_root, artifacts_root
    )
    by_id = {row["candidate_id"]: row for row in schedule_rows}
    selected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}

    for direction in sorted(grouped):
        existing = grouped[direction]
        accepted = sum(row["review_status"] == "accepted" for row in existing)
        rejected = len(existing) - accepted
        needed = TARGET_PER_DIRECTION - accepted
        final_rank = len(existing) + needed
        if final_rank > AVAILABLE_RANKS_PER_DIRECTION:
            raise RuntimeError(
                f"Not enough preregistered ranks remain for {direction}; "
                "stop and ask the user before changing the protocol."
            )
        counts[direction] = {
            "accepted": accepted,
            "rejected": rejected,
            "current_candidates": len(existing),
            "new_candidates_needed": needed,
        }
        for rank in range(len(existing) + 1, final_rank + 1):
            match = [
                row for row in schedule_rows
                if row["direction"] == direction
                and int(row["schedule_rank"]) == rank
            ]
            if len(match) != 1:
                raise RuntimeError(f"Scheduled replacement missing: {direction}, {rank}")
            candidate = match[0]
            if candidate["candidate_id"] not in by_id:
                raise RuntimeError("Invalid schedule entry.")
            selected.append({
                "candidate_id": candidate["candidate_id"],
                "direction": direction,
                "schedule_rank": rank,
            })

    if not selected:
        print("PASS: all ten directions already have 20 accepted images.")
        return
    if round_number == 1 and len(review_rows) != 200:
        raise RuntimeError("Round 1 must begin after the original 200-image review.")
    if round_number > 1:
        previous = plan_path(artifacts_root, round_number - 1)
        if not previous.is_file():
            raise RuntimeError("Previous replacement round is not recorded.")
    candidate_dir = artifacts_root / "cue_conflict/candidates"
    if any(
        (candidate_dir / f"{item['candidate_id']}.png").exists()
        for item in selected
    ):
        raise RuntimeError("A planned replacement image already exists.")

    snapshot_name = f"review_snapshot_round_{round_number:02d}.csv"
    snapshot_data = review_path.read_bytes()
    snapshot_paths = (
        artifacts_root / "cue_conflict/manifests" / snapshot_name,
        repo_root / "task1/results/cue_conflict" / snapshot_name,
    )
    for path in snapshot_paths:
        if path.exists() and path.read_bytes() != snapshot_data:
            raise RuntimeError(f"Existing review snapshot differs: {path}")
        if not path.exists():
            atomic_write(path, snapshot_data)

    plan = {
        "round": round_number,
        "purpose": "preregistered replacements for manually rejected AdaIN conflicts",
        "generation_device": device,
        "alpha": ALPHA,
        "schedule_file": str(schedule_path),
        "schedule_sha256": sha256_file(schedule_path),
        "review_manifest_before_generation": str(review_path),
        "review_manifest_sha256_before_generation": sha256_file(review_path),
        "review_snapshot_before_generation": str(snapshot_paths[0]),
        "review_snapshot_sha256": digest_bytes(snapshot_data),
        "review_rows_canonical_sha256_before_generation": digest_bytes(
            canonical_csv_bytes(review_rows, fields)
        ),
        "reviewed_candidates_before_generation": len(review_rows),
        "accepted_before_generation": sum(
            row["review_status"] == "accepted" for row in review_rows
        ),
        "rejected_before_generation": sum(
            row["review_status"] == "rejected" for row in review_rows
        ),
        "target_accepted_per_direction": TARGET_PER_DIRECTION,
        "counts_by_direction": counts,
        "planned_candidate_count": len(selected),
        "planned_candidates": selected,
        "planned_candidate_images_present_at_planning": 0,
        "selection_rule": "next unused ascending rank per deficient direction",
        "model_predictions_used_for_selection": False,
        "model_predictions_permitted_before_final_manual_review": False,
    }
    data = (json.dumps(plan, indent=2) + "\n").encode("utf-8")
    drive_path = plan_path(artifacts_root, round_number)
    repo_path = (
        repo_root / "task1/results/cue_conflict" / drive_path.name
    )
    for path in (drive_path, repo_path):
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f"Existing plan differs; refusing to overwrite {path}")
        if not path.exists():
            atomic_write(path, data)
    print(f"PASS: CPU replacement plan recorded before generation: {drive_path}")
    for direction, values in counts.items():
        print(f"{direction}: {values['new_candidates_needed']} replacements")
    print(f"Total planned replacements: {len(selected)}")
    print("PASS: no candidate images or model predictions generated by planning.")


def generation_row(
    schedule_row: dict[str, str],
    image: torch.Tensor,
    output_path: Path,
    round_number: int,
    device: str,
) -> dict[str, str]:
    return {
        **schedule_row,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_width": str(IMAGE_SIZE),
        "output_height": str(IMAGE_SIZE),
        "pixel_minimum": str(float(image.min())),
        "pixel_maximum": str(float(image.max())),
        "pixel_mean": str(float(image.mean())),
        "pixel_standard_deviation": str(float(image.std())),
        "model_predictions_used_for_generation": "False",
        "review_status": "pending",
        "rejection_reason": "",
        "reviewed_by": "",
        "review_completed_before_model_evaluation": "",
    }


def generate(
    repo_root: Path,
    artifacts_root: Path,
    data_root: Path,
    round_number: int,
    device: str,
    batch_size: int,
) -> None:
    if batch_size <= 0 or batch_size > GENERATION_BATCH_SIZE:
        raise ValueError("Batch size must be between 1 and 8.")
    plan_file = plan_path(artifacts_root, round_number)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if plan["round"] != round_number or plan["generation_device"] != device:
        raise RuntimeError("Requested round/device differs from the saved plan.")
    if plan["model_predictions_used_for_selection"] is not False:
        raise RuntimeError("Replacement selection used model predictions.")

    schedule_path = repo_root / "task1/results/cue_conflict/candidate_schedule.csv"
    if sha256_file(schedule_path) != plan["schedule_sha256"]:
        raise RuntimeError("Preregistered schedule changed since planning.")
    schedule_rows = read_csv(schedule_path)
    schedule = {row["candidate_id"]: row for row in schedule_rows}
    review_path = artifacts_root / "cue_conflict/manifests/review_manifest.csv"
    source_path = repo_root / "task1/results/cue_conflict/adain_source.json"
    fields, review_rows = read_review(review_path)
    selected_ids = [row["candidate_id"] for row in plan["planned_candidates"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Duplicate IDs in replacement plan.")
    selected_set = set(selected_ids)
    base_rows = [row for row in review_rows if row["candidate_id"] not in selected_set]
    generated_rows = [row for row in review_rows if row["candidate_id"] in selected_set]

    if len(base_rows) != plan["reviewed_candidates_before_generation"]:
        raise RuntimeError("Pre-generation review rows changed.")
    if digest_bytes(canonical_csv_bytes(base_rows, fields)) != plan[
        "review_rows_canonical_sha256_before_generation"
    ]:
        raise RuntimeError("Pre-generation review decisions changed.")
    base_grouped = validate_review(
        base_rows, schedule_rows, artifacts_root / "cue_conflict/candidates"
    )
    expected_ids = []
    for direction in sorted(base_grouped):
        existing = base_grouped[direction]
        accepted = sum(row["review_status"] == "accepted" for row in existing)
        needed = TARGET_PER_DIRECTION - accepted
        for rank in range(len(existing) + 1, len(existing) + needed + 1):
            matches = [
                row["candidate_id"] for row in schedule_rows
                if row["direction"] == direction
                and int(row["schedule_rank"]) == rank
            ]
            if len(matches) != 1:
                raise RuntimeError("Expected replacement is absent from schedule.")
            expected_ids.append(matches[0])
    if selected_ids != expected_ids:
        raise RuntimeError("Plan does not follow the next-unused-rank rule.")
    if [row["candidate_id"] for row in generated_rows] != selected_ids[:len(generated_rows)]:
        raise RuntimeError("Replacement generation is not a plan prefix.")
    if any(row["review_status"] != "pending" for row in generated_rows):
        raise RuntimeError("A replacement has already been reviewed; generation is locked.")
    candidate_dir = artifacts_root / "cue_conflict/candidates"
    for row in generated_rows:
        image_path = candidate_dir / f"{row['candidate_id']}.png"
        if not image_path.is_file() or sha256_file(image_path) != row["output_sha256"]:
            raise RuntimeError(f"Previously generated image changed: {image_path}")

    remaining_ids = selected_ids[len(generated_rows):]
    if not remaining_ids:
        print("PASS: all planned replacements are already generated.")
    else:
        for candidate_id in remaining_ids:
            image_path = candidate_dir / f"{candidate_id}.png"
            if image_path.exists():
                raise RuntimeError(
                    f"Unrecorded candidate image exists: {image_path}. "
                    "Stop rather than overwrite it."
                )

        seed_everything()
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        encoder, decoder, adain_function, source = load_adain(
            source_path, torch_device
        )
        dataset = STL10(root=str(data_root), split="test", download=False)

        with torch.inference_mode():
            for start in range(0, len(remaining_ids), batch_size):
                batch_ids = remaining_ids[start:start + batch_size]
                batch_schedule = [schedule[candidate_id] for candidate_id in batch_ids]
                content_images = []
                style_images = []
                for row in batch_schedule:
                    content, content_label = dataset[int(row["content_official_index"])]
                    style, style_label = dataset[int(row["style_official_index"])]
                    if int(content_label) != int(row["content_class_id"]):
                        raise RuntimeError("Content label mismatch.")
                    if int(style_label) != int(row["style_class_id"]):
                        raise RuntimeError("Style label mismatch.")
                    content_images.append(common_rgb_tensor(content))
                    style_images.append(common_rgb_tensor(style))

                content_batch = torch.stack(content_images).to(torch_device)
                style_batch = torch.stack(style_images).to(torch_device)
                content_features = encoder(content_batch)
                style_features = encoder(style_batch)
                transferred = adain_function.adaptive_instance_normalization(
                    content_features, style_features
                )
                blended = ALPHA * transferred + (1.0 - ALPHA) * content_features
                outputs = decoder(blended).clamp(0.0, 1.0).cpu()
                if tuple(outputs.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
                    raise RuntimeError("Unexpected AdaIN output shape.")
                if not bool(torch.isfinite(outputs).all()):
                    raise RuntimeError("AdaIN produced non-finite pixels.")

                for row, output in zip(batch_schedule, outputs):
                    candidate_id = row["candidate_id"]
                    image_path = candidate_dir / f"{candidate_id}.png"
                    save_png(output, image_path)
                    with Image.open(image_path) as image:
                        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                            raise RuntimeError("Saved PNG failed validation.")
                    review_rows.append(generation_row(
                        row, output, image_path, round_number, device
                    ))
                    atomic_write(review_path, canonical_csv_bytes(review_rows, fields))
                    print(f"Saved {candidate_id} ({len(review_rows) - len(base_rows)}/{len(selected_ids)})")

        del encoder, decoder, adain_function, source, dataset

    # A completed round has a compact generation record in Drive and Git.
    _, review_rows = read_review(review_path)
    replacements = [row for row in review_rows if row["candidate_id"] in selected_set]
    if [row["candidate_id"] for row in replacements] != selected_ids:
        raise RuntimeError("Replacement round is incomplete.")
    generation_fields = [field for field in fields if field not in (
        "review_status", "rejection_reason", "reviewed_by",
        "review_completed_before_model_evaluation",
    )]
    generation_rows = [
        {field: row[field] for field in generation_fields}
        for row in replacements
    ]
    manifest_dir = artifacts_root / "cue_conflict/manifests"
    basename = f"replacement_generation_round_{round_number:02d}"
    generation_manifest = manifest_dir / f"{basename}.csv"
    manifest_data = canonical_csv_bytes(generation_rows, generation_fields)
    if generation_manifest.exists():
        if generation_manifest.read_bytes() != manifest_data:
            raise RuntimeError("Existing generation manifest differs.")
    else:
        atomic_write(generation_manifest, manifest_data)
    metadata = {
        "round": round_number,
        "generation_device": device,
        "torch_version": str(torch.__version__),
        "batch_size": batch_size,
        "alpha": ALPHA,
        "generated_candidates": len(replacements),
        "source_partition": "official STL-10 test",
        "preregistered_schedule_sha256": plan["schedule_sha256"],
        "pre_generation_plan": str(plan_file),
        "pre_generation_plan_sha256": sha256_file(plan_file),
        "adain_source_record": str(source_path),
        "adain_source_record_sha256": sha256_file(source_path),
        "generation_manifest": str(generation_manifest),
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "review_status_at_generation": "all pending",
        "model_predictions_used_for_generation_or_selection": False,
        "model_predictions_run_before_manual_review": False,
    }
    metadata_path = manifest_dir / f"{basename}.json"
    metadata_data = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
    if metadata_path.exists():
        if metadata_path.read_bytes() != metadata_data:
            raise RuntimeError("Existing generation metadata differs.")
    else:
        atomic_write(metadata_path, metadata_data)
    result_dir = repo_root / "task1/results/cue_conflict"
    for source_file in (generation_manifest, metadata_path):
        target = result_dir / source_file.name
        if target.exists() and target.read_bytes() != source_file.read_bytes():
            raise RuntimeError(f"Existing repository result differs: {target}")
        if not target.exists():
            atomic_write(target, source_file.read_bytes())
    print(f"PASS: {len(replacements)} {device.upper()} AdaIN replacements generated.")
    print("PASS: all replacement review decisions remain pending.")
    print("PASS: generation device, schedule, weights, and image hashes recorded.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "generate"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.round < 1:
        raise ValueError("Round number must be positive.")
    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    if args.mode == "plan":
        create_plan(repo_root, artifacts_root, args.round, args.device)
        return
    if args.data_root is None:
        raise ValueError("--data-root is required in generate mode.")
    generate(
        repo_root, artifacts_root, args.data_root.resolve(),
        args.round, args.device, args.batch_size,
    )


if __name__ == "__main__":
    main()
