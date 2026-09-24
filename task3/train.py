"""Train one approved Task 3 configuration using source-only model selection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
from pathlib import Path

# Required by deterministic CUDA matrix multiplication when PyTorch requests it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from task3.config import (
    RUNS,
    code_tree_sha256,
    load_config,
    sha256_file,
    verify_preregistration,
)
from task3.data import (
    SEED,
    SOURCES,
    load_source_protocol,
    make_source_train_loader,
    make_validation_loader,
    steps_per_source_epoch,
    verify_source_snapshot,
)
from task3.methods.dan_dg import dan_dg_pairwise_mmd
from task3.methods.sam import clip_gradients, sam_step
from task3.model import (
    PACSClassifier,
    assert_batchnorm_unchanged,
    batchnorm_buffers,
    freeze_batchnorm_statistics,
    state_dict_sha256,
)


EXPECTED_SHARED_MMD_SHA256 = (
    "cfe0b1d9c22d7f492ea5e8f76732fbabf21c86cb53f24759af65fda09f9bfbcc"
)
EXPECTED_INITIALIZATION_STATE_SHA256 = (
    "4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3"
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def core_environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def lock_experiment(path: Path, identity: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != identity:
            raise RuntimeError(
                "Cross-run Task 3 identity differs from experiment_lock.json"
            )
        return
    atomic_write_json(identity, path)


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty training history")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_common_initialization(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("seed") != SEED:
        raise RuntimeError("Common initialization has the wrong seed")
    if payload.get("weights") != "ResNet18_Weights.IMAGENET1K_V1":
        raise RuntimeError("Common initialization has the wrong pretrained weights")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Common initialization has no state_dict")
    actual_hash = state_dict_sha256(state)
    if actual_hash != EXPECTED_INITIALIZATION_STATE_SHA256:
        raise RuntimeError("Common initialization state differs from Task 2")
    if payload.get("state_dict_sha256") != actual_hash:
        raise RuntimeError("Common initialization internal state hash is invalid")
    return state, actual_hash


def verify_code_preflight(
    path: Path,
    *,
    run_id: str,
    config: dict,
    code_sha256: str,
    protocol_sha256: str,
    source_snapshot_sha256: str,
    initialization_sha256: str,
    shared_mmd_sha256: str,
    preregistration_sha256: str,
    environment: dict,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Task 3 code-preflight record is missing: {path}")
    record = json.loads(path.read_text())
    expected_fields = {
        "status": "TASK3_CODE_PREFLIGHT_PASS",
        "training_started": False,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "code_tree_sha256": code_sha256,
        "source_protocol_sha256": protocol_sha256,
        "initialization_state_sha256": initialization_sha256,
        "preregistration_sha256": preregistration_sha256,
        "runtime": environment,
    }
    for name, expected in expected_fields.items():
        if record.get(name) != expected:
            raise RuntimeError(
                f"Code-preflight field {name!r} differs from this training run"
            )
    if record.get("configs", {}).get(run_id) != config:
        raise RuntimeError(f"Code preflight did not approve configuration {run_id}")
    if record.get("source_dataset", {}).get("source_snapshot_sha256") != (
        source_snapshot_sha256
    ):
        raise RuntimeError("Code preflight used a different source snapshot")
    if record.get("shared_mmd", {}).get("sha256") != shared_mmd_sha256:
        raise RuntimeError("Code preflight used a different shared MMD implementation")
    return sha256_file(path)


def validate_sources(
    model: PACSClassifier,
    loaders: dict,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}

    with torch.inference_mode():
        for domain in SOURCES:
            truth: list[int] = []
            predicted: list[int] = []
            for images, labels, identifiers in loaders[domain]:
                if any(not identifier.startswith(domain + "/") for identifier in identifiers):
                    raise RuntimeError(f"Validation loader crossed domains for {domain}")
                logits, _ = model(images.to(device, non_blocking=True))
                truth.extend(int(value) for value in labels.tolist())
                predicted.extend(
                    int(value) for value in logits.argmax(dim=1).cpu().tolist()
                )

            metrics[f"{domain}_accuracy"] = float(
                accuracy_score(truth, predicted)
            )
            metrics[f"{domain}_macro_f1"] = float(
                f1_score(
                    truth,
                    predicted,
                    labels=list(range(7)),
                    average="macro",
                    zero_division=0,
                )
            )

    metrics["mean_source_accuracy"] = float(
        np.mean([metrics[f"{domain}_accuracy"] for domain in SOURCES])
    )
    metrics["mean_source_macro_f1"] = float(
        np.mean([metrics[f"{domain}_macro_f1"] for domain in SOURCES])
    )
    metrics["worst_source_accuracy"] = float(
        min(metrics[f"{domain}_accuracy"] for domain in SOURCES)
    )
    metrics["worst_source_macro_f1"] = float(
        min(metrics[f"{domain}_macro_f1"] for domain in SOURCES)
    )
    return metrics


def make_epoch_source_loaders(
    protocol: dict,
    pacs_source_root: Path,
    epoch_index: int,
    steps: int,
    num_workers: int,
    pin_memory: bool,
) -> dict:
    loaders = {}
    for domain_index, domain in enumerate(SOURCES):
        base_seed = SEED + 100_000 * (domain_index + 1) + 10_000 * epoch_index
        loaders[domain] = make_source_train_loader(
            pacs_source_root,
            protocol["source_splits"][domain]["train"],
            steps,
            sampler_seed=base_seed,
            worker_seed=base_seed + 1,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def train_epoch(
    model: PACSClassifier,
    optimizer: torch.optim.Optimizer,
    protocol: dict,
    pacs_source_root: Path,
    config: dict,
    epoch_index: int,
    steps: int,
    device: torch.device,
    num_workers: int,
) -> dict[str, float]:
    model.train()
    freeze_batchnorm_statistics(model)

    source_loaders = make_epoch_source_loaders(
        protocol,
        pacs_source_root,
        epoch_index,
        steps,
        num_workers,
        pin_memory=device.type == "cuda",
    )
    source_iters = {domain: iter(loader) for domain, loader in source_loaders.items()}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    totals = {
        "classification_loss": 0.0,
        "sam_perturbed_classification_loss": 0.0,
        "mmd_loss": 0.0,
        "mmd_photo__art_painting": 0.0,
        "mmd_photo__cartoon": 0.0,
        "mmd_art_painting__cartoon": 0.0,
        "mmd_median_photo__art_painting": 0.0,
        "mmd_median_photo__cartoon": 0.0,
        "mmd_median_art_painting__cartoon": 0.0,
        "mmd_off_diagonal_zeros_photo__art_painting": 0.0,
        "mmd_off_diagonal_zeros_photo__cartoon": 0.0,
        "mmd_off_diagonal_zeros_art_painting__cartoon": 0.0,
        "sam_first_gradient_norm": 0.0,
        "sam_perturbation_norm": 0.0,
        "gradient_norm": 0.0,
        "gradient_norm_after_clipping": 0.0,
        "gradient_clipped_fraction": 0.0,
    }

    for _step_index in range(steps):
        source_batches = [next(source_iters[domain]) for domain in SOURCES]
        for domain, batch in zip(SOURCES, source_batches, strict=True):
            identifiers = batch[2]
            if len(identifiers) != 8:
                raise RuntimeError(f"{domain} batch does not contain eight examples")
            if any(not identifier.startswith(domain + "/") for identifier in identifiers):
                raise RuntimeError(f"Source batch crossed domains for {domain}")

        source_images = torch.cat([batch[0] for batch in source_batches], dim=0).to(
            device,
            non_blocking=True,
        )
        source_labels = torch.cat([batch[1] for batch in source_batches], dim=0).to(
            device,
            non_blocking=True,
        )
        if len(source_labels) != 24:
            raise RuntimeError("A Task 3 update does not contain 24 source examples")

        if config["method"] == "dan_dg":
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(source_images)
            classification_loss = nn.functional.cross_entropy(
                logits,
                source_labels,
                weight=None,
                reduction=config["class_loss_reduction"],
                label_smoothing=config["label_smoothing"],
            )
            mmd_loss, pair_diagnostics = dan_dg_pairwise_mmd(
                features,
                batch_per_domain=config["source_batch_per_domain"],
            )
            total_loss = classification_loss + config["mmd_lambda"] * mmd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("DAN-DG total loss is non-finite")
            total_loss.backward()
            gradient_norm, post_clip_norm, was_clipped = clip_gradients(
                parameters,
                max_norm=config["gradient_clipping"],
                norm_type=config["gradient_clipping_norm_type"],
                foreach=config["gradient_clipping_foreach"],
            )
            optimizer.step()

            totals["classification_loss"] += float(
                classification_loss.detach().item()
            )
            totals["mmd_loss"] += float(mmd_loss.detach().item())
            for details in pair_diagnostics:
                totals[f"mmd_{details.pair}"] += details.mmd_squared
                totals[f"mmd_median_{details.pair}"] += (
                    details.median_squared_distance
                )
                totals[f"mmd_off_diagonal_zeros_{details.pair}"] += float(
                    details.off_diagonal_zero_count
                )
            totals["gradient_norm"] += gradient_norm
            totals["gradient_norm_after_clipping"] += post_clip_norm
            totals["gradient_clipped_fraction"] += float(was_clipped)

        elif config["method"] == "sam":

            def loss_closure() -> torch.Tensor:
                logits, _ = model(source_images)
                return nn.functional.cross_entropy(
                    logits,
                    source_labels,
                    weight=None,
                    reduction=config["class_loss_reduction"],
                    label_smoothing=config["label_smoothing"],
                )

            diagnostics = sam_step(
                model,
                optimizer,
                loss_closure,
                rho=config["sam_rho"],
                max_gradient_norm=config["gradient_clipping"],
            )
            totals["classification_loss"] += diagnostics.first_loss
            totals["sam_perturbed_classification_loss"] += (
                diagnostics.perturbed_loss
            )
            totals["sam_first_gradient_norm"] += diagnostics.first_gradient_norm
            totals["sam_perturbation_norm"] += diagnostics.perturbation_norm
            totals["gradient_norm"] += diagnostics.second_gradient_norm
            totals["gradient_norm_after_clipping"] += (
                diagnostics.second_gradient_norm_after_clipping
            )
            totals["gradient_clipped_fraction"] += float(
                diagnostics.second_gradient_was_clipped
            )
        else:
            raise RuntimeError(f"Unexpected Task 3 method: {config['method']}")

    return {name: value / steps for name, value in totals.items()}


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(args.run_id)
    if args.num_workers != config["num_workers"]:
        raise RuntimeError(
            f"num_workers is locked to {config['num_workers']}; got {args.num_workers}"
        )
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The locked Task 3 training environment requires CUDA")

    device = torch.device("cuda")
    actual_environment = core_environment(device)
    if actual_environment != EXPECTED_RUNTIME:
        raise RuntimeError(
            "Training environment differs from the locked Task 2/3 runtime.\n"
            f"Expected: {EXPECTED_RUNTIME}\nActual:   {actual_environment}"
        )

    protocol = load_source_protocol(args.protocol)
    dataset_summary = verify_source_snapshot(args.pacs_source_root, protocol)
    preregistration_sha256 = verify_preregistration(args.preregistration)

    shared_mmd_path = root / "shared" / "mmd.py"
    shared_mmd_sha256 = sha256_file(shared_mmd_path)
    if shared_mmd_sha256 != EXPECTED_SHARED_MMD_SHA256:
        raise RuntimeError("The locked Task 2 MMD implementation changed")

    seed_everything(SEED)
    code_sha256 = code_tree_sha256(root)
    protocol_sha256 = sha256_file(args.protocol)
    initialization_state, initialization_sha256 = load_common_initialization(
        args.initialization
    )
    code_preflight_sha256 = verify_code_preflight(
        args.code_preflight,
        run_id=args.run_id,
        config=config,
        code_sha256=code_sha256,
        protocol_sha256=protocol_sha256,
        source_snapshot_sha256=dataset_summary["source_snapshot_sha256"],
        initialization_sha256=initialization_sha256,
        shared_mmd_sha256=shared_mmd_sha256,
        preregistration_sha256=preregistration_sha256,
        environment=actual_environment,
    )

    model = PACSClassifier(pretrained=False)
    model.load_state_dict(initialization_state)
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
    if steps != config["steps_per_source_epoch"]:
        raise RuntimeError(f"Unexpected source steps per epoch: {steps}")

    resume_identity = {
        "config": config,
        "protocol_sha256": protocol_sha256,
        "source_snapshot_sha256": dataset_summary["source_snapshot_sha256"],
        "initialization_sha256": initialization_sha256,
        "shared_mmd_sha256": shared_mmd_sha256,
        "code_sha256": code_sha256,
        "environment": actual_environment,
        "preregistration_sha256": preregistration_sha256,
        "code_preflight_sha256": code_preflight_sha256,
        "num_workers": args.num_workers,
        "source_only_phase": True,
        "sketch_images_accessed": 0,
    }
    shared_experiment_identity = {
        key: value for key, value in resume_identity.items() if key != "config"
    }

    args.output.mkdir(parents=True, exist_ok=True)
    lock_experiment(args.output / "experiment_lock.json", shared_experiment_identity)

    run_dir = args.output / args.run_id
    if args.resume:
        if not (run_dir / "last.pt").is_file():
            raise FileNotFoundError(f"No completed-epoch checkpoint in {run_dir}")
    else:
        run_dir.mkdir(exist_ok=False)

    history: list[dict] = []
    best_f1 = float("-inf")
    best_epoch: int | None = None
    stale_epochs = 0
    start_epoch = 0

    if args.resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved.get("identity") != resume_identity:
            raise RuntimeError("Resume identity differs from the saved Task 3 state")
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

        training_metrics = train_epoch(
            model,
            optimizer,
            protocol,
            args.pacs_source_root,
            config,
            epoch_index,
            steps,
            device,
            args.num_workers,
        )
        assert_batchnorm_unchanged(model, original_batchnorm)
        validation_metrics = validate_sources(model, validation_loaders, device)

        row = {"epoch": epoch_index + 1, **training_metrics, **validation_metrics}
        history.append(row)

        current_f1 = validation_metrics["mean_source_macro_f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch_index + 1
            stale_epochs = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": best_epoch,
                    "source_validation": validation_metrics,
                    "identity": resume_identity,
                    "source_only_phase": True,
                    "sketch_images_accessed": 0,
                },
                run_dir / "best.pt",
            )
            atomic_write_json(
                {"epoch": best_epoch, **validation_metrics},
                run_dir / "best_source_validation.json",
            )
        else:
            stale_epochs += 1

        write_history(run_dir / "history.csv", history)
        atomic_torch_save(
            {
                "identity": resume_identity,
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
                    **training_metrics,
                    **validation_metrics,
                    "sketch_images_accessed": 0,
                }
            ),
            flush=True,
        )

    if best_epoch is None:
        raise RuntimeError("Task 3 training produced no selectable checkpoint")

    manifest = {
        "status": "TASK3_RUN_COMPLETE",
        "run_id": args.run_id,
        "identity": resume_identity,
        "steps_per_source_epoch": steps,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_mean_source_macro_f1": best_f1,
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "source_only_phase": True,
        "sketch_images_accessed": 0,
    }
    atomic_write_json(manifest, run_dir / "run.json")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=tuple(RUNS), required=True)
    parser.add_argument("--pacs-source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--code-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
