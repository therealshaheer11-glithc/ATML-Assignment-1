"""Calibrate the approved source-only initialization bandwidth floors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from shared.mmd import l2_normalize_alignment_features
from task3.config import code_tree_sha256, sha256_file
from task3.data import (
    SEED,
    SOURCES,
    CyclingBatchSampler,
    SourceDataset,
    load_source_protocol,
    steps_per_source_epoch,
    torch_generator,
    verify_source_snapshot,
)
from task3.model import (
    PACSClassifier,
    assert_batchnorm_unchanged,
    batchnorm_buffers,
)
from task3.research_variants.bandwidth_floor import (
    PAIR_NAMES,
    current_pair_median,
    reduce_calibration_medians,
)
from task3.research_variants.config import (
    VARIANT_ID,
    VARIANT_PROTOCOL_VERSION,
    verify_research_protocol,
)
from task3.train import (
    EXPECTED_INITIALIZATION_STATE_SHA256,
    EXPECTED_RUNTIME,
    atomic_write_json,
    core_environment,
    load_common_initialization,
    seed_everything,
)


def load_authorization(path: Path, code_sha256: str, protocol_sha256: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Research-variant authorization is missing: {path}")
    record = json.loads(path.read_text())
    expected = {
        "status": "TASK3_BANDWIDTH_FLOOR_VARIANT_AUTHORIZED",
        "variant_id": VARIANT_ID,
        "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
        "source_only_phase": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
        "code_tree_sha256": code_sha256,
        "source_protocol_sha256": protocol_sha256,
        "initialization_state_sha256": EXPECTED_INITIALIZATION_STATE_SHA256,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise RuntimeError(f"Research authorization field differs: {name}")
    return record


def make_calibration_loaders(
    root: Path,
    protocol: dict,
    *,
    steps: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, DataLoader]:
    """Use epoch-zero sampler seeds with deterministic center-crop datasets."""
    loaders: dict[str, DataLoader] = {}
    for domain_index, domain in enumerate(SOURCES):
        base_seed = SEED + 100_000 * (domain_index + 1)
        records = protocol["source_splits"][domain]["train"]
        loaders[domain] = DataLoader(
            SourceDataset(root, records, training=False),
            batch_sampler=CyclingBatchSampler(
                len(records), batch_size=8, steps=steps, seed=base_seed
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=torch_generator(base_seed + 1),
        )
    return loaders


def run(args: argparse.Namespace) -> None:
    actual_root = Path(__file__).resolve().parents[1]
    if args.code_root.resolve() != actual_root:
        raise RuntimeError("--code-root does not identify the imported repository")
    if args.output.exists():
        raise FileExistsError(f"Bandwidth floors already exist: {args.output}")
    if args.num_workers != 2:
        raise RuntimeError("Calibration num_workers is locked to 2")
    if not torch.cuda.is_available():
        raise RuntimeError("The calibrated research variant requires CUDA")

    device = torch.device("cuda")
    environment = core_environment(device)
    if environment != EXPECTED_RUNTIME:
        raise RuntimeError(
            f"Calibration runtime differs. Expected {EXPECTED_RUNTIME}, got {environment}"
        )

    protocol = load_source_protocol(args.protocol)
    dataset = verify_source_snapshot(args.pacs_source_root, protocol)
    if dataset.get("sketch_images_accessed") != 0:
        raise RuntimeError("Source verification reports Sketch access")
    steps = steps_per_source_epoch(protocol)
    if steps != 235:
        raise RuntimeError(f"Unexpected calibration step count: {steps}")

    code_sha256 = code_tree_sha256(actual_root)
    protocol_sha256 = sha256_file(args.protocol)
    research_protocol_sha256 = verify_research_protocol(args.research_protocol)
    authorization = load_authorization(
        args.authorization, code_sha256, protocol_sha256
    )

    initialization, initialization_sha256 = load_common_initialization(
        args.initialization
    )
    if initialization_sha256 != EXPECTED_INITIALIZATION_STATE_SHA256:
        raise RuntimeError("Calibration initialization differs")

    seed_everything(SEED)
    model = PACSClassifier(pretrained=False)
    model.load_state_dict(initialization, strict=True)
    model.to(device)
    model.eval()
    original_batchnorm = batchnorm_buffers(model)

    loaders = make_calibration_loaders(
        args.pacs_source_root,
        protocol,
        steps=steps,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    iterators = {domain: iter(loader) for domain, loader in loaders.items()}
    medians: dict[str, list[float]] = {pair: [] for pair in PAIR_NAMES}
    selected_identifier_digest = hashlib.sha256()
    unique_identifiers: dict[str, set[str]] = {domain: set() for domain in SOURCES}

    with torch.inference_mode():
        for step_index in range(steps):
            batches = [next(iterators[domain]) for domain in SOURCES]
            for domain, batch in zip(SOURCES, batches, strict=True):
                identifiers = [str(value) for value in batch[2]]
                if len(identifiers) != 8:
                    raise RuntimeError("Calibration batch size differs")
                if any(not value.startswith(domain + "/") for value in identifiers):
                    raise RuntimeError("Calibration loader crossed source domains")
                unique_identifiers[domain].update(identifiers)
                for identifier in identifiers:
                    selected_identifier_digest.update(
                        f"{step_index}\t{domain}\t{identifier}\n".encode("utf-8")
                    )

            images = torch.cat([batch[0] for batch in batches], dim=0).to(
                device, non_blocking=True
            )
            _logits, features = model(images)
            normalized = tuple(
                l2_normalize_alignment_features(block)
                for block in features.split(8, dim=0)
            )
            pair_indices = ((0, 1), (0, 2), (1, 2))
            for pair, (left, right) in zip(PAIR_NAMES, pair_indices, strict=True):
                value, pair_count, _zeros = current_pair_median(
                    normalized[left], normalized[right]
                )
                if pair_count != 120:
                    raise RuntimeError("Calibration pair did not contain 120 distances")
                medians[pair].append(float(value.item()))

    assert_batchnorm_unchanged(model, original_batchnorm)
    reduced = reduce_calibration_medians(medians)
    result = {
        "status": "TASK3_BANDWIDTH_FLOOR_CALIBRATION_PASS",
        "variant_id": VARIANT_ID,
        "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
        "research_variant": True,
        "primary_protocol_replacement": False,
        "source_only_phase": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
        "class_labels_used_in_calibration_computation": False,
        "calibration": {
            "model_state": "authenticated_common_initialization_before_training",
            "data": "source_training_records_only",
            "transform": "resize_256_center_crop_224_imagenet_normalization",
            "sampler": "exact_epoch_zero_domain_sampler_seeds",
            "steps": steps,
            "batch_per_domain": 8,
            "pair_distance_candidates_per_step": 120,
            "reduction": "median_of_235_current_batch_pair_medians",
            "floor_multiplier": 1.0,
            "identifier_sequence_sha256": selected_identifier_digest.hexdigest(),
            "unique_identifiers_per_domain": {
                domain: len(unique_identifiers[domain]) for domain in SOURCES
            },
        },
        "pair_floors": reduced,
        "pair_batch_medians": medians,
        "identities": {
            "repository_commit": authorization["repository_commit"],
            "code_tree_sha256": code_sha256,
            "source_protocol_sha256": protocol_sha256,
            "source_snapshot_sha256": dataset["source_snapshot_sha256"],
            "initialization_state_sha256": initialization_sha256,
            "research_protocol_sha256": research_protocol_sha256,
            "authorization_sha256": sha256_file(args.authorization),
            "runtime": environment,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result, args.output)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--pacs-source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--research-protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
