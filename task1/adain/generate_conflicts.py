"""Prepare and render the preregistered Task 1 AdaIN cue conflicts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torchvision.datasets import STL10


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from task1.data.make_subset import STL10_CLASS_NAMES  # noqa: E402
from task1.transforms import (  # noqa: E402
    common_rgb_tensor,
    common_tensor_to_pil,
)


SEED = 6304
ALPHA = 0.8
INITIAL_RANKS_PER_DIRECTION = 20
AVAILABLE_RANKS_PER_DIRECTION = 50
GENERATION_BATCH_SIZE = 8
IMAGE_SIZE = 224
PAIRS = (
    ("airplane", "cat"),
    ("car", "deer"),
    ("ship", "dog"),
    ("truck", "horse"),
    ("bird", "monkey"),
)
EXPECTED_ADAIN_COMMIT_PREFIX = "324eede"
SCHEDULE_FIELDS = (
    "candidate_id",
    "pair_number",
    "pair_name",
    "schedule_rank",
    "direction_code",
    "direction",
    "content_official_index",
    "content_class_id",
    "content_class_name",
    "style_official_index",
    "style_class_id",
    "style_class_name",
    "shape_class_id",
    "shape_class_name",
    "texture_class_id",
    "texture_class_name",
    "selection_seed",
    "alpha",
    "part_of_initial_generation",
)
REJECTION_REASONS = (
    "content object is no longer visually recognizable",
    "object is severely obscured or structurally corrupted",
    "output contains blank regions or major rendering artifacts",
    "no perceptible style or texture transfer occurred",
)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def csv_text(rows: list[dict[str, Any]], fields: Iterable[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: Iterable[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    selected_fields = list(fields) if fields is not None else list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text(rows, selected_fields), encoding="utf-8")


def update_pairing_configuration(configuration_path: Path) -> None:
    text = configuration_path.read_text(encoding="utf-8")
    marker = "    generate_additional_candidates_until_target_met: true\n"
    existing_marker = "    pairing_protocol:\n"

    if existing_marker in text:
        print("PASS: cue-conflict pairing protocol already recorded.")
        return

    if marker not in text:
        raise RuntimeError(
            "Could not safely locate the cue-conflict pairing section."
        )

    replacement = marker + """    pairing_protocol:
      source: "fixed balanced 500-image official-test evaluation subset"
      schedule_seed: 6304
      schedule_construction: "deterministic one-to-one class-pair matching"
      available_matched_pairs_per_class_pair: 50
      bidirectional_from_same_matched_pair: true
      initial_schedule_ranks_per_direction: [1, 20]
      replacement_rule: "next unused ascending schedule rank in deficient direction"
      candidate_selection_may_not_use_model_predictions: true
      approved_by_user_before_generation: true
      schedule_file: "task1/results/cue_conflict/candidate_schedule.csv"
"""
    configuration_path.write_text(
        text.replace(marker, replacement, 1),
        encoding="utf-8",
    )
    print("PASS: cue-conflict pairing protocol recorded in task1.yaml.")


def evaluation_indices_by_class(
    evaluation_manifest: Path,
) -> dict[str, list[int]]:
    rows = read_csv(evaluation_manifest)
    if len(rows) != 500:
        raise RuntimeError("Expected exactly 500 evaluation-manifest rows.")

    by_class: dict[str, list[int]] = {
        name: [] for name in STL10_CLASS_NAMES
    }
    seen_indices: set[int] = set()

    for row in rows:
        if row["dataset"] != "STL-10":
            raise RuntimeError("Unexpected dataset in evaluation manifest.")
        if row["official_partition"] != "test":
            raise RuntimeError("Cue conflicts must use official test images.")

        index = int(row["official_index"])
        class_id = int(row["class_id"])
        class_name = row["class_name"]

        if index in seen_indices:
            raise RuntimeError("Duplicate evaluation image identifier.")
        if STL10_CLASS_NAMES[class_id] != class_name:
            raise RuntimeError("Evaluation class ID/name mismatch.")

        seen_indices.add(index)
        by_class[class_name].append(index)

    if any(len(indices) != 50 for indices in by_class.values()):
        raise RuntimeError("Expected exactly 50 evaluation images per class.")

    return by_class


def make_schedule(
    evaluation_manifest: Path,
) -> list[dict[str, Any]]:
    by_class = evaluation_indices_by_class(evaluation_manifest)
    class_to_id = {
        name: class_id
        for class_id, name in enumerate(STL10_CLASS_NAMES)
    }
    rows: list[dict[str, Any]] = []

    for pair_index, (class_a, class_b) in enumerate(PAIRS, start=1):
        seed_sequence = np.random.SeedSequence([SEED, pair_index])
        generator = np.random.default_rng(seed_sequence)
        indices_a = generator.permutation(by_class[class_a]).tolist()
        indices_b = generator.permutation(by_class[class_b]).tolist()

        for zero_based_rank, (index_a, index_b) in enumerate(
            zip(indices_a, indices_b),
        ):
            rank = zero_based_rank + 1
            base = {
                "pair_number": pair_index,
                "pair_name": f"{class_a}__{class_b}",
                "schedule_rank": rank,
                "selection_seed": SEED,
                "alpha": ALPHA,
                "part_of_initial_generation": (
                    rank <= INITIAL_RANKS_PER_DIRECTION
                ),
            }

            rows.append(
                {
                    "candidate_id": (
                        f"p{pair_index:02d}_r{rank:02d}_a_to_b"
                    ),
                    **base,
                    "direction_code": "a_to_b",
                    "direction": f"{class_a}_shape__{class_b}_texture",
                    "content_official_index": int(index_a),
                    "content_class_id": class_to_id[class_a],
                    "content_class_name": class_a,
                    "style_official_index": int(index_b),
                    "style_class_id": class_to_id[class_b],
                    "style_class_name": class_b,
                    "shape_class_id": class_to_id[class_a],
                    "shape_class_name": class_a,
                    "texture_class_id": class_to_id[class_b],
                    "texture_class_name": class_b,
                }
            )
            rows.append(
                {
                    "candidate_id": (
                        f"p{pair_index:02d}_r{rank:02d}_b_to_a"
                    ),
                    **base,
                    "direction_code": "b_to_a",
                    "direction": f"{class_b}_shape__{class_a}_texture",
                    "content_official_index": int(index_b),
                    "content_class_id": class_to_id[class_b],
                    "content_class_name": class_b,
                    "style_official_index": int(index_a),
                    "style_class_id": class_to_id[class_a],
                    "style_class_name": class_a,
                    "shape_class_id": class_to_id[class_b],
                    "shape_class_name": class_b,
                    "texture_class_id": class_to_id[class_a],
                    "texture_class_name": class_a,
                }
            )

    if len(rows) != 500:
        raise RuntimeError("Expected a full schedule of 500 candidates.")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise RuntimeError("Candidate IDs are not unique.")

    direction_counts = Counter(row["direction"] for row in rows)
    if set(direction_counts.values()) != {50} or len(direction_counts) != 10:
        raise RuntimeError("Expected 50 scheduled candidates per direction.")

    return rows


def prepare_schedule(
    repo_root: Path,
    artifacts_root: Path,
) -> tuple[Path, Path]:
    configuration_path = repo_root / "task1/configs/task1.yaml"
    evaluation_manifest = (
        repo_root / "task1/results/splits/evaluation_subset.csv"
    )
    source_record = (
        repo_root / "task1/results/cue_conflict/adain_source.json"
    )
    schedule_path = (
        repo_root / "task1/results/cue_conflict/candidate_schedule.csv"
    )
    schedule_record_path = (
        repo_root / "task1/results/cue_conflict/candidate_schedule.json"
    )

    for required_path in (
        configuration_path,
        evaluation_manifest,
        source_record,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    source = json.loads(source_record.read_text(encoding="utf-8"))
    if not source["approved_by_user_before_generation"]:
        raise RuntimeError("AdaIN source was not approved before generation.")
    if not source["executable_implementation"]["commit"].startswith(
        EXPECTED_ADAIN_COMMIT_PREFIX
    ):
        raise RuntimeError("Unexpected AdaIN implementation commit.")

    update_pairing_configuration(configuration_path)
    schedule = make_schedule(evaluation_manifest)
    expected_text = csv_text(schedule, SCHEDULE_FIELDS)

    if schedule_path.exists():
        if schedule_path.read_bytes() != expected_text.encode("utf-8"):
            raise RuntimeError(
                "Existing candidate schedule differs from the approved rule."
            )
        print("PASS: existing candidate schedule matches approved rule.")
    else:
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_path.write_text(expected_text, encoding="utf-8")
        print("PASS: wrote deterministic 500-candidate schedule.")

    candidate_directory = artifacts_root / "cue_conflict/candidates"
    current_image_count = (
        len(list(candidate_directory.glob("*.png")))
        if candidate_directory.exists()
        else 0
    )

    if schedule_record_path.exists():
        schedule_record = json.loads(
            schedule_record_path.read_text(encoding="utf-8")
        )
        if schedule_record["schedule_sha256"] != sha256_file(schedule_path):
            raise RuntimeError("Candidate schedule checksum changed.")
        if schedule_record["candidate_images_present_at_schedule_creation"] != 0:
            raise RuntimeError("Schedule was not recorded before generation.")
        print("PASS: existing pre-generation schedule record verified.")
    else:
        if current_image_count != 0:
            raise RuntimeError(
                "Candidate images already exist before schedule registration."
            )
        schedule_record = {
            "dataset": "STL-10",
            "source_subset": (
                "fixed balanced 500-image official-test evaluation subset"
            ),
            "selection_seed": SEED,
            "class_pairs": [list(pair) for pair in PAIRS],
            "directions_per_pair": 2,
            "available_matched_pairs_per_class_pair": 50,
            "total_scheduled_candidates": 500,
            "initial_ranks_per_direction": [1, 20],
            "initial_candidate_count": 200,
            "replacement_rule": (
                "next unused ascending schedule rank in deficient direction"
            ),
            "bidirectional_from_same_matched_pair": True,
            "model_predictions_used_for_schedule": False,
            "approved_by_user_before_generation": True,
            "candidate_images_present_at_schedule_creation": 0,
            "evaluation_manifest": str(evaluation_manifest),
            "evaluation_manifest_sha256": sha256_file(
                evaluation_manifest
            ),
            "adain_source_record": str(source_record),
            "adain_source_record_sha256": sha256_file(source_record),
            "schedule_file": str(schedule_path),
            "schedule_sha256": sha256_file(schedule_path),
        }
        schedule_record_path.write_text(
            json.dumps(schedule_record, indent=2) + "\n",
            encoding="utf-8",
        )
        print("PASS: recorded schedule before candidate generation.")

    return schedule_path, schedule_record_path


def load_external_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_adain(
    source_record_path: Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, Any, dict[str, Any]]:
    source = json.loads(source_record_path.read_text(encoding="utf-8"))
    implementation = source["executable_implementation"]
    if not implementation["commit"].startswith(
        EXPECTED_ADAIN_COMMIT_PREFIX
    ):
        raise RuntimeError("Unexpected AdaIN implementation commit.")

    external_repository = Path(implementation["local_checkout"])
    models_directory = external_repository / "models"

    for filename, details in source["pretrained_weights"].items():
        weight_path = models_directory / filename
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)
        if sha256_file(weight_path) != details["sha256"]:
            raise RuntimeError(f"AdaIN weight checksum mismatch: {filename}")

    sys.path.insert(0, str(external_repository))
    adain_function = load_external_module(
        "function",
        external_repository / "function.py",
    )
    adain_net = load_external_module(
        "pinned_adain_net",
        external_repository / "net.py",
    )

    decoder = adain_net.decoder
    vgg = adain_net.vgg
    decoder.load_state_dict(
        torch.load(
            models_directory / "decoder.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    vgg.load_state_dict(
        torch.load(
            models_directory / "vgg_normalised.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    encoder = nn.Sequential(*list(vgg.children())[:31])
    encoder.requires_grad_(False).eval().to(device)
    decoder.requires_grad_(False).eval().to(device)
    return encoder, decoder, adain_function, source


def batched(rows: list[dict[str, str]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def save_png(image: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.png")
    common_tensor_to_pil(image).save(temporary_path, format="PNG")
    os.replace(temporary_path, path)


def generate_initial_candidates(
    repo_root: Path,
    data_root: Path,
    artifacts_root: Path,
    device: torch.device,
) -> None:
    schedule_path, schedule_record_path = prepare_schedule(
        repo_root,
        artifacts_root,
    )
    schedule_record = json.loads(
        schedule_record_path.read_text(encoding="utf-8")
    )
    if schedule_record["candidate_images_present_at_schedule_creation"] != 0:
        raise RuntimeError("Invalid pre-generation schedule record.")

    generation_directory = artifacts_root / "cue_conflict/manifests"
    review_manifest = generation_directory / "review_manifest.csv"
    existing_review: list[dict[str, str]] | None = None
    if review_manifest.exists():
        existing_review = read_csv(review_manifest)
        if any(
            row["review_status"] != "pending"
            for row in existing_review
        ):
            raise RuntimeError(
                "Refusing to regenerate candidates after manual review began."
            )

    schedule = read_csv(schedule_path)
    selected_rows = [
        row
        for row in schedule
        if row["part_of_initial_generation"] == "True"
    ]
    if len(selected_rows) != 200:
        raise RuntimeError("Expected exactly 200 initial candidates.")

    source_record_path = (
        repo_root / "task1/results/cue_conflict/adain_source.json"
    )
    encoder, decoder, adain_function, source = load_adain(
        source_record_path,
        device,
    )
    dataset = STL10(root=str(data_root), split="test", download=False)
    candidate_directory = artifacts_root / "cue_conflict/candidates"
    generated_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch_number, batch_rows in enumerate(
            batched(selected_rows, GENERATION_BATCH_SIZE),
            start=1,
        ):
            content_images = []
            style_images = []

            for row in batch_rows:
                content_index = int(row["content_official_index"])
                style_index = int(row["style_official_index"])
                content_image, content_label = dataset[content_index]
                style_image, style_label = dataset[style_index]

                if int(content_label) != int(row["content_class_id"]):
                    raise RuntimeError("Content label mismatch.")
                if int(style_label) != int(row["style_class_id"]):
                    raise RuntimeError("Style label mismatch.")

                content_images.append(common_rgb_tensor(content_image))
                style_images.append(common_rgb_tensor(style_image))

            content_batch = torch.stack(content_images).to(device)
            style_batch = torch.stack(style_images).to(device)
            content_features = encoder(content_batch)
            style_features = encoder(style_batch)
            transferred_features = (
                adain_function.adaptive_instance_normalization(
                    content_features,
                    style_features,
                )
            )
            blended_features = (
                ALPHA * transferred_features
                + (1.0 - ALPHA) * content_features
            )
            outputs = decoder(blended_features).clamp(0.0, 1.0).cpu()

            if tuple(outputs.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(
                    f"Unexpected AdaIN output shape: {tuple(outputs.shape)}"
                )
            if not bool(torch.isfinite(outputs).all()):
                raise RuntimeError("AdaIN generated non-finite pixels.")

            for row, output in zip(batch_rows, outputs):
                output_path = (
                    candidate_directory / f"{row['candidate_id']}.png"
                )
                save_png(output, output_path)
                generated_rows.append(
                    {
                        **row,
                        "output_path": str(output_path),
                        "output_sha256": sha256_file(output_path),
                        "output_width": IMAGE_SIZE,
                        "output_height": IMAGE_SIZE,
                        "pixel_minimum": float(output.min()),
                        "pixel_maximum": float(output.max()),
                        "pixel_mean": float(output.mean()),
                        "pixel_standard_deviation": float(output.std()),
                        "model_predictions_used_for_generation": False,
                    }
                )

            print(
                f"Generated batch {batch_number}/"
                f"{(len(selected_rows) + GENERATION_BATCH_SIZE - 1) // GENERATION_BATCH_SIZE}"
            )

    if len(generated_rows) != 200:
        raise RuntimeError("Initial generation did not produce 200 images.")
    if len({row["candidate_id"] for row in generated_rows}) != 200:
        raise RuntimeError("Generated candidate IDs are not unique.")

    generation_directory.mkdir(parents=True, exist_ok=True)
    generation_manifest = (
        generation_directory / "initial_generation_manifest.csv"
    )
    write_csv(generation_manifest, generated_rows)

    review_fields = list(generated_rows[0]) + [
        "review_status",
        "rejection_reason",
        "reviewed_by",
        "review_completed_before_model_evaluation",
    ]
    pending_review_rows = [
        {
            **row,
            "review_status": "pending",
            "rejection_reason": "",
            "reviewed_by": "",
            "review_completed_before_model_evaluation": "",
        }
        for row in generated_rows
    ]

    if existing_review is not None:
        if [row["candidate_id"] for row in existing_review] != [
            row["candidate_id"] for row in pending_review_rows
        ]:
            raise RuntimeError("Existing review manifest has different IDs.")
    else:
        write_csv(review_manifest, pending_review_rows, review_fields)

    metadata = {
        "dataset": "STL-10",
        "source_partition": "official test",
        "source_subset": "fixed balanced 500-image evaluation subset",
        "selection_seed": SEED,
        "alpha": ALPHA,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "image_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "initial_candidate_count": 200,
        "initial_candidates_per_direction": 20,
        "direction_count": 10,
        "model_predictions_used_for_generation": False,
        "model_predictions_run_before_manual_review": False,
        "schedule_file": str(schedule_path),
        "schedule_sha256": sha256_file(schedule_path),
        "schedule_record": str(schedule_record_path),
        "schedule_record_sha256": sha256_file(schedule_record_path),
        "adain_source_record": str(source_record_path),
        "adain_source_record_sha256": sha256_file(source_record_path),
        "adain_commit": source["executable_implementation"]["commit"],
        "weight_sha256": {
            name: details["sha256"]
            for name, details in source["pretrained_weights"].items()
        },
        "generation_manifest": str(generation_manifest),
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "review_manifest": str(review_manifest),
        "review_status_at_generation": "all pending",
        "rejection_reasons": list(REJECTION_REASONS),
    }
    metadata_path = generation_directory / "generation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    create_contact_sheets(
        generated_rows=generated_rows,
        dataset=dataset,
        artifacts_root=artifacts_root,
    )

    repository_result_directory = (
        repo_root / "task1/results/cue_conflict"
    )
    repository_result_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        generation_manifest,
        repository_result_directory / generation_manifest.name,
    )
    shutil.copy2(
        metadata_path,
        repository_result_directory / metadata_path.name,
    )

    print("PASS: generated exactly 200 initial cue-conflict candidates.")
    print(f"Review manifest: {review_manifest}")
    print(
        "PASS: every candidate remains pending manual review; "
        "no classifier was loaded or evaluated."
    )


def create_contact_sheets(
    generated_rows: list[dict[str, Any]],
    dataset: STL10,
    artifacts_root: Path,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generated_rows:
        grouped[str(row["direction"])].append(row)

    sheet_directory = artifacts_root / "cue_conflict/review/contact_sheets"
    sheet_directory.mkdir(parents=True, exist_ok=True)
    image_side = 160
    tile_padding = 8
    tile_width = image_side * 3 + tile_padding * 2
    tile_height = image_side + 45
    columns = 4
    rows_per_sheet = 5

    for direction, rows in grouped.items():
        rows.sort(key=lambda row: int(row["schedule_rank"]))
        if len(rows) != INITIAL_RANKS_PER_DIRECTION:
            raise RuntimeError(
                f"Expected 20 initial candidates for {direction}."
            )

        canvas = Image.new(
            "RGB",
            (
                columns * tile_width,
                60 + rows_per_sheet * tile_height,
            ),
            color="white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 8),
            f"{direction} | C = content/shape, O = output, S = style/texture",
            fill="black",
        )
        draw.text(
            (10, 30),
            "Manual visual review only — do not use model predictions",
            fill="black",
        )

        for position, row in enumerate(rows):
            grid_row = position // columns
            grid_column = position % columns
            x = grid_column * tile_width
            y = 60 + grid_row * tile_height
            content_image, _ = dataset[
                int(row["content_official_index"])
            ]
            style_image, _ = dataset[int(row["style_official_index"])]
            content_pil = common_tensor_to_pil(
                common_rgb_tensor(content_image)
            ).resize((image_side, image_side), Image.Resampling.BICUBIC)
            style_pil = common_tensor_to_pil(
                common_rgb_tensor(style_image)
            ).resize((image_side, image_side), Image.Resampling.BICUBIC)
            with Image.open(row["output_path"]) as output_image:
                output_pil = output_image.convert("RGB").resize(
                    (image_side, image_side),
                    Image.Resampling.BICUBIC,
                )

            canvas.paste(content_pil, (x, y))
            canvas.paste(output_pil, (x + image_side, y))
            canvas.paste(style_pil, (x + 2 * image_side, y))
            draw.rectangle(
                (x, y, x + tile_width - 1, y + tile_height - 1),
                outline="gray",
                width=1,
            )
            draw.text(
                (x + 4, y + image_side + 3),
                f"{row['candidate_id']} | C       O       S",
                fill="black",
            )

        sheet_path = sheet_directory / f"{direction}.png"
        canvas.save(sheet_path, format="PNG")

    if len(list(sheet_directory.glob("*.png"))) != 10:
        raise RuntimeError("Expected exactly 10 direction contact sheets.")
    print(f"PASS: wrote 10 manual-review contact sheets to {sheet_directory}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prepare", "generate-initial"),
        required=True,
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    seed_everything()

    if args.mode == "prepare":
        prepare_schedule(repo_root, artifacts_root)
        print("PASS: schedule preparation finished with zero generated images.")
        return

    if args.data_root is None:
        raise ValueError("--data-root is required for generate-initial mode.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    generate_initial_candidates(
        repo_root=repo_root,
        data_root=args.data_root.resolve(),
        artifacts_root=artifacts_root,
        device=device,
    )


if __name__ == "__main__":
    main()
