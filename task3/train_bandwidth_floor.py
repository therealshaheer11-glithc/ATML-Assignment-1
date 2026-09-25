"""Train the approved supplementary DAN-DG bandwidth-floor study."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from task3.config import code_tree_sha256, sha256_file
from task3.data import (
    SEED,
    SOURCES,
    load_source_protocol,
    make_validation_loader,
    steps_per_source_epoch,
    verify_source_snapshot,
)
from task3.methods.sam import clip_gradients
from task3.model import (
    PACSClassifier,
    assert_batchnorm_unchanged,
    batchnorm_buffers,
    freeze_batchnorm_statistics,
)
from task3.research_variants.bandwidth_floor import (
    PAIR_NAMES,
    dan_dg_pairwise_mmd_with_floor,
)
from task3.research_variants.config import (
    VARIANT_ID,
    VARIANT_PROTOCOL_VERSION,
    VARIANT_RUNS,
    load_variant_config,
    verify_research_protocol,
)
from task3.train import (
    EXPECTED_INITIALIZATION_STATE_SHA256,
    EXPECTED_RUNTIME,
    EXPECTED_SHARED_MMD_SHA256,
    atomic_torch_save,
    atomic_write_json,
    core_environment,
    load_common_initialization,
    lock_experiment,
    make_epoch_source_loaders,
    seed_everything,
    validate_sources,
    write_history,
)


def load_calibration(
    path: Path,
    *,
    code_sha256: str,
    protocol_sha256: str,
    source_snapshot_sha256: str,
    research_protocol_sha256: str,
    environment: dict,
) -> tuple[dict[str, float], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Bandwidth-floor calibration is missing: {path}")
    record = json.loads(path.read_text())
    expected = {
        "status": "TASK3_BANDWIDTH_FLOOR_CALIBRATION_PASS",
        "variant_id": VARIANT_ID,
        "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
        "research_variant": True,
        "primary_protocol_replacement": False,
        "source_only_phase": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
        "class_labels_used_in_calibration_computation": False,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise RuntimeError(f"Calibration field differs: {name}")
    identities = record.get("identities", {})
    expected_identities = {
        "code_tree_sha256": code_sha256,
        "source_protocol_sha256": protocol_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "initialization_state_sha256": EXPECTED_INITIALIZATION_STATE_SHA256,
        "research_protocol_sha256": research_protocol_sha256,
        "runtime": environment,
    }
    for name, value in expected_identities.items():
        if identities.get(name) != value:
            raise RuntimeError(f"Calibration identity differs: {name}")

    calibration = record.get("calibration", {})
    if calibration.get("steps") != 235 or calibration.get("batch_per_domain") != 8:
        raise RuntimeError("Calibration sampling design differs")
    if calibration.get("floor_multiplier") != 1.0:
        raise RuntimeError("Calibration introduced an unapproved floor multiplier")

    pair_records = record.get("pair_floors", {})
    if set(pair_records) != set(PAIR_NAMES):
        raise RuntimeError("Calibration does not contain all three pair floors")
    floors: dict[str, float] = {}
    for pair in PAIR_NAMES:
        details = pair_records[pair]
        value = float(details["floor_squared_distance"])
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(f"Invalid frozen floor for {pair}")
        if details.get("sample_count") != 235 or details.get("floor_multiplier") != 1.0:
            raise RuntimeError(f"Calibration reduction differs for {pair}")
        if value != float(details.get("median")):
            raise RuntimeError(f"Frozen floor is not the recorded median for {pair}")
        floors[pair] = value
    return floors, record


def train_epoch(
    model: PACSClassifier,
    optimizer: torch.optim.Optimizer,
    protocol: dict,
    pacs_source_root: Path,
    config: dict,
    floors: dict[str, float],
    epoch_index: int,
    steps: int,
    device: torch.device,
    num_workers: int,
) -> dict[str, float]:
    model.train()
    freeze_batchnorm_statistics(model)
    loaders = make_epoch_source_loaders(
        protocol,
        pacs_source_root,
        epoch_index,
        steps,
        num_workers,
        pin_memory=device.type == "cuda",
    )
    iterators = {domain: iter(loader) for domain, loader in loaders.items()}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    totals: dict[str, float] = {
        "classification_loss": 0.0,
        "mmd_loss": 0.0,
        "mmd_floor_activation_fraction": 0.0,
        "gradient_norm": 0.0,
        "gradient_norm_after_clipping": 0.0,
        "gradient_clipped_fraction": 0.0,
    }
    for pair in PAIR_NAMES:
        totals.update(
            {
                f"mmd_{pair}": 0.0,
                f"mmd_current_median_{pair}": 0.0,
                f"mmd_effective_median_{pair}": 0.0,
                f"mmd_frozen_floor_{pair}": 0.0,
                f"mmd_floor_activation_fraction_{pair}": 0.0,
                f"mmd_off_diagonal_zeros_{pair}": 0.0,
            }
        )

    for _step_index in range(steps):
        batches = [next(iterators[domain]) for domain in SOURCES]
        for domain, batch in zip(SOURCES, batches, strict=True):
            identifiers = batch[2]
            if len(identifiers) != 8:
                raise RuntimeError(f"{domain} batch does not contain eight examples")
            if any(not value.startswith(domain + "/") for value in identifiers):
                raise RuntimeError(f"Source batch crossed domains for {domain}")

        images = torch.cat([batch[0] for batch in batches], dim=0).to(
            device, non_blocking=True
        )
        labels = torch.cat([batch[1] for batch in batches], dim=0).to(
            device, non_blocking=True
        )
        if len(labels) != 24:
            raise RuntimeError("A variant update does not contain 24 source examples")

        optimizer.zero_grad(set_to_none=True)
        logits, features = model(images)
        classification = nn.functional.cross_entropy(
            logits,
            labels,
            weight=None,
            reduction=config["class_loss_reduction"],
            label_smoothing=config["label_smoothing"],
        )
        mmd, diagnostics = dan_dg_pairwise_mmd_with_floor(
            features,
            floors,
            batch_per_domain=config["source_batch_per_domain"],
        )
        total_loss = classification + config["mmd_lambda"] * mmd
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Bandwidth-floor total loss is non-finite")
        total_loss.backward()
        gradient_norm, post_clip_norm, was_clipped = clip_gradients(
            parameters,
            max_norm=config["gradient_clipping"],
            norm_type=config["gradient_clipping_norm_type"],
            foreach=config["gradient_clipping_foreach"],
        )
        optimizer.step()

        totals["classification_loss"] += float(classification.detach().item())
        totals["mmd_loss"] += float(mmd.detach().item())
        totals["gradient_norm"] += gradient_norm
        totals["gradient_norm_after_clipping"] += post_clip_norm
        totals["gradient_clipped_fraction"] += float(was_clipped)
        active_count = 0
        for details in diagnostics:
            pair = details.pair
            active_count += int(details.floor_activated)
            totals[f"mmd_{pair}"] += details.mmd_squared
            totals[f"mmd_current_median_{pair}"] += (
                details.current_median_squared_distance
            )
            totals[f"mmd_effective_median_{pair}"] += (
                details.effective_median_squared_distance
            )
            totals[f"mmd_frozen_floor_{pair}"] += (
                details.frozen_floor_squared_distance
            )
            totals[f"mmd_floor_activation_fraction_{pair}"] += float(
                details.floor_activated
            )
            totals[f"mmd_off_diagonal_zeros_{pair}"] += float(
                details.off_diagonal_zero_count
            )
        totals["mmd_floor_activation_fraction"] += active_count / len(PAIR_NAMES)

    return {name: value / steps for name, value in totals.items()}


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_variant_config(args.run_id)
    if args.num_workers != config["num_workers"]:
        raise RuntimeError("Variant num_workers differs from the primary protocol")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The variant training environment requires CUDA")

    device = torch.device("cuda")
    environment = core_environment(device)
    if environment != EXPECTED_RUNTIME:
        raise RuntimeError(f"Variant runtime differs: {environment}")
    protocol = load_source_protocol(args.protocol)
    dataset = verify_source_snapshot(args.pacs_source_root, protocol)
    if dataset.get("sketch_images_accessed") != 0:
        raise RuntimeError("Source snapshot verification reports Sketch access")

    shared_mmd_path = root / "shared" / "mmd.py"
    if sha256_file(shared_mmd_path) != EXPECTED_SHARED_MMD_SHA256:
        raise RuntimeError("The locked Task 2/3 MMD implementation changed")
    protocol_sha256 = sha256_file(args.protocol)
    code_sha256 = code_tree_sha256(root)
    research_protocol_sha256 = verify_research_protocol(args.research_protocol)
    floors, calibration = load_calibration(
        args.calibration,
        code_sha256=code_sha256,
        protocol_sha256=protocol_sha256,
        source_snapshot_sha256=dataset["source_snapshot_sha256"],
        research_protocol_sha256=research_protocol_sha256,
        environment=environment,
    )
    initialization, initialization_sha256 = load_common_initialization(
        args.initialization
    )

    seed_everything(SEED)
    model = PACSClassifier(pretrained=False)
    model.load_state_dict(initialization, strict=True)
    model.to(device)
    original_batchnorm = batchnorm_buffers(model)
    seed_everything(SEED)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config["adam_betas"]),
        eps=config["adam_epsilon"],
        amsgrad=config["amsgrad"],
        foreach=config["adam_foreach"],
        fused=config["adam_fused"],
        maximize=False,
        capturable=False,
        differentiable=False,
    )
    validation_loaders = {
        domain: make_validation_loader(
            args.pacs_source_root,
            protocol["source_splits"][domain]["validation"],
            batch_size=config["validation_batch_size"],
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for domain in SOURCES
    }
    steps = steps_per_source_epoch(protocol)
    if steps != 235:
        raise RuntimeError(f"Unexpected source step count: {steps}")

    identity = {
        "config": config,
        "variant_id": VARIANT_ID,
        "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256,
        "source_snapshot_sha256": dataset["source_snapshot_sha256"],
        "initialization_sha256": initialization_sha256,
        "shared_mmd_sha256": sha256_file(shared_mmd_path),
        "code_sha256": code_sha256,
        "environment": environment,
        "research_protocol_sha256": research_protocol_sha256,
        "calibration_sha256": sha256_file(args.calibration),
        "frozen_pair_floors": floors,
        "num_workers": args.num_workers,
        "research_variant": True,
        "primary_protocol_replacement": False,
        "source_only_phase": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
    }
    shared_identity = {key: value for key, value in identity.items() if key != "config"}
    args.output.mkdir(parents=True, exist_ok=True)
    lock_experiment(args.output / "experiment_lock.json", shared_identity)

    run_dir = args.output / args.run_id
    if args.resume:
        if not (run_dir / "last.pt").is_file():
            raise FileNotFoundError(f"No resumable checkpoint in {run_dir}")
    else:
        run_dir.mkdir(exist_ok=False)

    history: list[dict] = []
    best_f1 = float("-inf")
    best_epoch: int | None = None
    stale_epochs = 0
    start_epoch = 0
    if args.resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise RuntimeError("Resume identity differs from the saved variant state")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history = saved["history"]
        best_f1 = saved["best_f1"]
        best_epoch = saved["best_epoch"]
        stale_epochs = saved["stale_epochs"]
        start_epoch = saved["completed_epochs"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        assert_batchnorm_unchanged(model, original_batchnorm)

    for epoch_index in range(start_epoch, config["epochs"]):
        if stale_epochs >= config["patience"]:
            break
        training = train_epoch(
            model,
            optimizer,
            protocol,
            args.pacs_source_root,
            config,
            floors,
            epoch_index,
            steps,
            device,
            args.num_workers,
        )
        assert_batchnorm_unchanged(model, original_batchnorm)
        validation = validate_sources(model, validation_loaders, device)
        row = {"epoch": epoch_index + 1, **training, **validation}
        history.append(row)

        current_f1 = validation["mean_source_macro_f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch_index + 1
            stale_epochs = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": best_epoch,
                    "source_validation": validation,
                    "identity": identity,
                    "research_variant": True,
                    "primary_protocol_replacement": False,
                    "source_only_phase": True,
                    "final_evaluation_started": False,
                    "sketch_images_accessed": 0,
                },
                run_dir / "best.pt",
            )
            atomic_write_json(
                {"epoch": best_epoch, **validation},
                run_dir / "best_source_validation.json",
            )
        else:
            stale_epochs += 1

        write_history(run_dir / "history.csv", history)
        atomic_torch_save(
            {
                "identity": identity,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "completed_epochs": epoch_index + 1,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "research_variant": True,
                "source_only_phase": True,
                "sketch_images_accessed": 0,
            },
            run_dir / "last.pt",
        )
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "epoch": epoch_index + 1,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                    **training,
                    **validation,
                    "sketch_images_accessed": 0,
                }
            ),
            flush=True,
        )

    if best_epoch is None:
        raise RuntimeError("Variant training produced no selectable checkpoint")
    manifest = {
        "status": "TASK3_BANDWIDTH_FLOOR_RUN_COMPLETE",
        "run_id": args.run_id,
        "method": "dan_dg_bandwidth_floor",
        "mmd_lambda": config["mmd_lambda"],
        "identity": identity,
        "steps_per_source_epoch": steps,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_mean_source_macro_f1": best_f1,
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "calibration_sha256": sha256_file(args.calibration),
        "research_variant": True,
        "primary_protocol_replacement": False,
        "source_only_phase": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
    }
    atomic_write_json(manifest, run_dir / "run.json")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=tuple(VARIANT_RUNS), required=True)
    parser.add_argument("--pacs-source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--research-protocol", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
