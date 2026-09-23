"""Train one locked Task 2 configuration using source-only checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from shared.mmd import l2_normalize_alignment_features, three_kernel_mmd
from shared.pacs import (
    SEED,
    SOURCES,
    load_protocol,
    make_source_train_loader,
    make_target_train_loader,
    make_validation_loader,
    sha256_file,
    steps_per_source_epoch,
    verify_dataset_snapshot,
)
from task2.config import RUNS, load_config
from task2.methods import DomainDiscriminator, conditional_features, grl_strength, reverse_gradient
from task2.model import (
    PACSClassifier,
    assert_batchnorm_unchanged,
    batchnorm_buffers,
    freeze_batchnorm_statistics,
    state_dict_sha256,
)


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


def hash_code_tree(root: Path) -> str:
    digest = hashlib.sha256()
    selected: list[Path] = []
    for subdir in (root / "shared", root / "task2"):
        for path in subdir.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".md", ".txt"}:
                selected.append(path)
    for path in sorted(set(selected)):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def lock_experiment(path: Path, identity: dict) -> None:
    """Create once, then require the same shared identity for every method."""
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise RuntimeError(
                "Cross-run identity differs from the existing experiment_lock.json"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(identity, indent=2) + "\n")
    os.replace(temporary, path)


def write_history(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def verify_preregistration(path: Path) -> str:
    text = path.read_text().strip()
    if not text or "PENDING" in text:
        raise RuntimeError(
            "The student's DAN-strength expectation is still pending. "
            "Lock it before corrected training."
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_common_initialization(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("seed") != SEED or payload.get("weights") != "ResNet18_Weights.IMAGENET1K_V1":
        raise RuntimeError("Common initialization metadata is invalid")
    state = payload["state_dict"]
    actual_hash = state_dict_sha256(state)
    if actual_hash != payload.get("state_dict_sha256"):
        raise RuntimeError("Common initialization state hash is invalid")
    return state, actual_hash


def validate_sources(model: PACSClassifier, loaders: dict, device: torch.device) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}
    with torch.inference_mode():
        for domain in SOURCES:
            truth: list[int] = []
            predicted: list[int] = []
            for images, labels, _ in loaders[domain]:
                logits, _ = model(images.to(device, non_blocking=True))
                truth.extend(int(value) for value in labels.tolist())
                predicted.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())
            metrics[f"{domain}_accuracy"] = float(accuracy_score(truth, predicted))
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
    return metrics


def total_gradient_norm(parameters: list[nn.Parameter]) -> float:
    squared_total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("A non-finite gradient was produced")
        squared_total += float(gradient.float().square().sum().item())
    norm = math.sqrt(squared_total)
    if not math.isfinite(norm):
        raise FloatingPointError("Total gradient norm is non-finite")
    return norm


def clip_gradients(
    parameters: list[nn.Parameter],
    max_norm: float,
    norm_type: float,
    foreach: bool,
) -> tuple[float, float, bool]:
    """Apply the locked global clipping policy and return pre/post diagnostics."""
    if max_norm <= 0 or norm_type != 2.0 or foreach is not False:
        raise ValueError("Unexpected gradient-clipping configuration")
    pre_clip_norm = total_gradient_norm(parameters)
    torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=True,
        foreach=foreach,
    )
    post_clip_norm = total_gradient_norm(parameters)
    tolerance = max_norm * 1e-5
    if post_clip_norm > max_norm + tolerance:
        raise FloatingPointError(
            f"Gradient clipping failed: {post_clip_norm} > {max_norm}"
        )
    return pre_clip_norm, post_clip_norm, pre_clip_norm > max_norm


def train_epoch(
    model: PACSClassifier,
    discriminator: DomainDiscriminator | None,
    optimizer: torch.optim.Optimizer,
    protocol: dict,
    pacs_root: Path,
    config: dict,
    epoch_index: int,
    steps: int,
    device: torch.device,
    num_workers: int,
) -> dict[str, float]:
    model.train()
    freeze_batchnorm_statistics(model)
    if discriminator is not None:
        discriminator.train()

    pin_memory = device.type == "cuda"
    source_loaders = {}
    for domain_index, domain in enumerate(SOURCES):
        base_seed = SEED + 100_000 * (domain_index + 1) + 10_000 * epoch_index
        source_loaders[domain] = make_source_train_loader(
            pacs_root,
            protocol["source_splits"][domain]["train"],
            steps,
            sampler_seed=base_seed,
            worker_seed=base_seed + 1,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    source_iters = {domain: iter(loader) for domain, loader in source_loaders.items()}

    target_iter = None
    if config["method"] != "source_only":
        base_seed = SEED + 900_000 + 10_000 * epoch_index
        target_loader = make_target_train_loader(
            pacs_root,
            protocol["target_unlabeled"],
            steps,
            sampler_seed=base_seed,
            worker_seed=base_seed + 1,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        target_iter = iter(target_loader)

    parameters = list(model.parameters())
    if discriminator is not None:
        parameters.extend(discriminator.parameters())

    totals = {
        "classification_loss": 0.0,
        "mmd_loss": 0.0,
        "mmd_median_squared_distance": 0.0,
        "mmd_off_diagonal_zero_count": 0.0,
        "domain_loss": 0.0,
        "domain_accuracy": 0.0,
        "grl_strength": 0.0,
        "adversarial_feature_norm_before": 0.0,
        "adversarial_feature_norm_after": 0.0,
        "gradient_norm": 0.0,
        "gradient_norm_after_clipping": 0.0,
        "gradient_clipped_fraction": 0.0,
    }
    planned_updates = config["epochs"] * steps

    for step_index in range(steps):
        source_batches = [next(source_iters[domain]) for domain in SOURCES]
        source_images = torch.cat([batch[0] for batch in source_batches], dim=0).to(
            device, non_blocking=True
        )
        source_labels = torch.cat([batch[1] for batch in source_batches], dim=0).to(
            device, non_blocking=True
        )
        if len(source_labels) != 24:
            raise RuntimeError("A source update does not contain 8 examples per domain")

        optimizer.zero_grad(set_to_none=True)
        if target_iter is None:
            source_logits, _ = model(source_images)
            logits = features = None
        else:
            target_images, _ = next(target_iter)
            if len(target_images) != 24:
                raise RuntimeError("An adaptation update does not contain 24 targets")
            combined_images = torch.cat(
                (source_images, target_images.to(device, non_blocking=True)), dim=0
            )
            logits, features = model(combined_images)
            source_logits = logits[:24]

        classification_loss = nn.functional.cross_entropy(
            source_logits,
            source_labels,
            weight=None,
            reduction=config["class_loss_reduction"],
            label_smoothing=config["label_smoothing"],
        )
        total_loss = classification_loss
        mmd_loss = classification_loss.new_zeros(())
        domain_loss = classification_loss.new_zeros(())
        domain_accuracy = 0.0
        current_grl = 0.0
        mmd_median = 0.0
        mmd_zeros = 0
        adversarial_norm_before = 0.0
        adversarial_norm_after = 0.0

        if config["method"] == "dan":
            assert features is not None
            if config["mmd_feature_normalization"] != "l2_per_sample":
                raise RuntimeError("Unexpected DAN MMD feature-normalization policy")
            source_mmd_features = l2_normalize_alignment_features(features[:24])
            target_mmd_features = l2_normalize_alignment_features(features[24:])
            mmd_loss, diagnostics = three_kernel_mmd(
                source_mmd_features, target_mmd_features
            )
            total_loss = total_loss + config["mmd_lambda"] * mmd_loss
            mmd_median = diagnostics.median_squared_distance
            mmd_zeros = diagnostics.off_diagonal_zero_count
        elif config["method"] in {"dann", "cdan"}:
            assert features is not None and logits is not None and discriminator is not None
            if config["adversarial_feature_normalization"] != "l2_per_sample":
                raise RuntimeError(
                    "Unexpected DANN/CDAN adversarial feature-normalization policy"
                )
            global_update = epoch_index * steps + step_index
            progress = global_update / max(1, planned_updates - 1)
            current_grl = grl_strength(progress)
            adversarial_features = l2_normalize_alignment_features(features)
            adversarial_norm_before = float(
                torch.linalg.vector_norm(
                    features.detach().float(), ord=2, dim=1
                ).mean().item()
            )
            adversarial_norm_after = float(
                torch.linalg.vector_norm(
                    adversarial_features.detach(), ord=2, dim=1
                ).mean().item()
            )
            domain_input = (
                conditional_features(adversarial_features, logits)
                if config["method"] == "cdan"
                else adversarial_features
            )
            domain_logits = discriminator(reverse_gradient(domain_input, current_grl))
            domain_labels = torch.cat(
                (
                    torch.zeros(24, dtype=torch.long, device=device),
                    torch.ones(24, dtype=torch.long, device=device),
                )
            )
            domain_loss = nn.functional.cross_entropy(
                domain_logits,
                domain_labels,
                weight=None,
                reduction=config["domain_loss_reduction"],
                label_smoothing=0.0,
            )
            total_loss = total_loss + config["domain_loss_weight"] * domain_loss
            domain_accuracy = float(
                (domain_logits.argmax(dim=1) == domain_labels).float().mean().item()
            )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch_index + 1}, step {step_index + 1}"
            )
        total_loss.backward()
        gradient_norm, gradient_norm_after_clipping, was_clipped = clip_gradients(
            parameters,
            max_norm=config["gradient_clipping"],
            norm_type=config["gradient_clipping_norm_type"],
            foreach=config["gradient_clipping_foreach"],
        )
        optimizer.step()

        totals["classification_loss"] += float(classification_loss.detach().item())
        totals["mmd_loss"] += float(mmd_loss.detach().item())
        totals["mmd_median_squared_distance"] += mmd_median
        totals["mmd_off_diagonal_zero_count"] += mmd_zeros
        totals["domain_loss"] += float(domain_loss.detach().item())
        totals["domain_accuracy"] += domain_accuracy
        totals["grl_strength"] += current_grl
        totals["adversarial_feature_norm_before"] += adversarial_norm_before
        totals["adversarial_feature_norm_after"] += adversarial_norm_after
        totals["gradient_norm"] += gradient_norm
        totals["gradient_norm_after_clipping"] += gradient_norm_after_clipping
        totals["gradient_clipped_fraction"] += float(was_clipped)

    return {name: value / steps for name, value in totals.items()}


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(args.run_id)
    if args.num_workers != config["num_workers"]:
        raise RuntimeError(
            f"num_workers is locked to {config['num_workers']}; got {args.num_workers}"
        )
    protocol = load_protocol(args.protocol)
    dataset_summary = verify_dataset_snapshot(args.pacs_root, protocol)
    preregistration_sha256 = verify_preregistration(args.preregistration)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    seed_everything(SEED)
    environment = core_environment(device)
    code_sha256 = hash_code_tree(root)
    protocol_sha256 = sha256_file(args.protocol)
    initialization_state, initialization_sha256 = load_common_initialization(
        args.initialization
    )

    # Constructing an untrained shell may consume RNG; load the common state and then
    # reset the seed before any method-specific discriminator initialization.
    model = PACSClassifier(pretrained=False)
    model.load_state_dict(initialization_state)
    model.to(device)
    original_bn = batchnorm_buffers(model)
    seed_everything(SEED)
    discriminator = None
    if config["method"] in {"dann", "cdan"}:
        discriminator = DomainDiscriminator(config["method"] == "cdan").to(device)
    # Make the stochastic training stream independent of the number of random values
    # consumed while constructing differently shaped method-specific modules.
    seed_everything(SEED)

    parameters = list(model.parameters())
    if discriminator is not None:
        parameters.extend(discriminator.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
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
            args.pacs_root,
            protocol["source_splits"][domain]["validation"],
            batch_size=config["validation_batch_size"],
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for domain in SOURCES
    }
    steps = steps_per_source_epoch(protocol)
    history: list[dict] = []
    best_f1 = float("-inf")
    best_epoch: int | None = None
    stale_epochs = 0
    start_epoch = 0

    resume_identity = {
        "config": config,
        "protocol_sha256": protocol_sha256,
        "initialization_sha256": initialization_sha256,
        "code_sha256": code_sha256,
        "environment": environment,
        "dataset_file_list_sha256": dataset_summary["file_list_sha256"],
        "preregistration_sha256": preregistration_sha256,
        "num_workers": args.num_workers,
    }
    shared_experiment_identity = {
        key: value for key, value in resume_identity.items() if key != "config"
    }
    args.output.mkdir(parents=True, exist_ok=True)
    lock_experiment(args.output / "experiment_lock.json", shared_experiment_identity)

    run_dir = args.output / args.run_id
    if args.resume:
        if not (run_dir / "last.pt").exists():
            raise FileNotFoundError(f"No completed-epoch checkpoint in {run_dir}")
    else:
        run_dir.mkdir(exist_ok=False)

    if args.resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved.get("identity") != resume_identity:
            raise RuntimeError("Resume identity differs from the completed-epoch checkpoint")
        model.load_state_dict(saved["model_state"])
        if discriminator is not None:
            discriminator.load_state_dict(saved["discriminator_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history = saved["history"]
        best_f1 = saved["best_f1"]
        best_epoch = saved["best_epoch"]
        stale_epochs = saved["stale_epochs"]
        start_epoch = saved["completed_epochs"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        assert_batchnorm_unchanged(model, original_bn)

    for epoch_index in range(start_epoch, config["epochs"]):
        if stale_epochs >= config["patience"]:
            break
        training_metrics = train_epoch(
            model,
            discriminator,
            optimizer,
            protocol,
            args.pacs_root,
            config,
            epoch_index,
            steps,
            device,
            args.num_workers,
        )
        assert_batchnorm_unchanged(model, original_bn)
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
                    "discriminator_state": (
                        discriminator.state_dict() if discriminator is not None else None
                    ),
                    "epoch": best_epoch,
                    "source_validation": validation_metrics,
                    "identity": resume_identity,
                    "target_labels_used": False,
                },
                run_dir / "best.pt",
            )
            (run_dir / "best_source_validation.json").write_text(
                json.dumps({"epoch": best_epoch, **validation_metrics}, indent=2) + "\n"
            )
        else:
            stale_epochs += 1

        write_history(run_dir / "history.csv", history)
        last_payload = {
            "identity": resume_identity,
            "model_state": model.state_dict(),
            "discriminator_state": (
                discriminator.state_dict() if discriminator is not None else None
            ),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "completed_epochs": epoch_index + 1,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
            "target_labels_used": False,
        }
        atomic_torch_save(last_payload, run_dir / "last.pt")
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "epoch": epoch_index + 1,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                    **training_metrics,
                    **validation_metrics,
                }
            ),
            flush=True,
        )

    if best_epoch is None:
        raise RuntimeError("Training produced no selectable checkpoint")
    manifest = {
        "run_id": args.run_id,
        "identity": resume_identity,
        "dataset_source": args.dataset_source,
        "steps_per_source_epoch": steps,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_mean_source_macro_f1": best_f1,
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "target_labels_used": False,
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "RUN_COMPLETE", **manifest}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=tuple(RUNS), required=True)
    parser.add_argument("--pacs-root", type=Path, required=True)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("shared/splits/pacs_sketch_seed6304.json"),
    )
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
