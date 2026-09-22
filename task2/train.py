"""Train a Task 2 configuration using only source validation for selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from shared.mmd import three_kernel_mmd
from shared.pacs import (
    PACSImages, SOURCES, SEED, CyclingBatchSampler, image_paths,
    load_protocol, resolve_root, steps_per_source_epoch,
)
from task2.methods import DomainDiscriminator, conditional_features, reversal_strength, reverse_gradient
from task2.model import PACSClassifier, assert_batchnorm_unchanged, batchnorm_buffers, freeze_batchnorm_statistics


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
MAIN_RUNS = {
    "source_only": ("source_only", None),
    "dan_0p1": ("dan", 0.1),
    "dan_1": ("dan", 1.0),
    "dan_10": ("dan", 10.0),
    "dann": ("dann", None),
    "cdan": ("cdan", None),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(run_id: str) -> dict:
    if run_id not in MAIN_RUNS:
        raise ValueError(f"Choose a locked run ID from {tuple(MAIN_RUNS)}")
    config = json.loads((CONFIG_DIR / "base.json").read_text())
    config.update(json.loads((CONFIG_DIR / f"{run_id}.json").read_text()))
    method, strength = MAIN_RUNS[run_id]
    if (config["run_id"], config["method"], config.get("mmd_lambda")) != (run_id, method, strength):
        raise ValueError("Run configuration does not match the registered experiment")
    expected = {
        "seed": 6304, "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "epochs": 30, "patience": 5, "source_batch_per_domain": 8,
        "target_batch": 24, "optimizer": "AdamW", "learning_rate": 1e-4,
        "weight_decay": 1e-4, "selection_metric": "mean_source_validation_macro_f1",
        "domain_loss_weight": 1.0, "grl_maximum": 1.0,
        "mmd_kernel_factors": [0.5, 1.0, 2.0],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("A fixed assignment setting was changed in the configuration")
    return config


def hash_model(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_dataset_snapshot(root: Path, protocol: dict) -> None:
    digest = hashlib.sha256()
    for domain in (*SOURCES, "sketch"):
        for path in image_paths(root / domain):
            digest.update(f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}\n".encode())
    if digest.hexdigest() != protocol["file_list_sha256"]:
        raise RuntimeError("PACS files differ from the recorded split snapshot")


def make_train_loader(root: Path, rows: list[dict], labeled: bool, batch_size: int, steps: int, seed: int, workers: int, device: torch.device) -> DataLoader:
    dataset = PACSImages(root, rows, training=True, labeled=labeled)
    return DataLoader(
        dataset,
        batch_sampler=CyclingBatchSampler(len(dataset), batch_size, steps, seed),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )


def make_validation_loader(root: Path, rows: list[dict], workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        PACSImages(root, rows, training=False, labeled=True),
        batch_size=64, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def validate_sources(model: PACSClassifier, loaders: dict[str, DataLoader], device: torch.device) -> dict[str, float]:
    model.eval()
    results = {}
    with torch.inference_mode():
        for domain in SOURCES:
            truth, predicted = [], []
            for images, labels, _ in loaders[domain]:
                logits, _ = model(images.to(device, non_blocking=True))
                truth.extend(labels.tolist())
                predicted.extend(logits.argmax(dim=1).cpu().tolist())
            results[f"{domain}_accuracy"] = float(accuracy_score(truth, predicted))
            results[f"{domain}_macro_f1"] = float(f1_score(truth, predicted, labels=list(range(7)), average="macro", zero_division=0))
    results["mean_source_accuracy"] = float(np.mean([results[f"{d}_accuracy"] for d in SOURCES]))
    results["mean_source_macro_f1"] = float(np.mean([results[f"{d}_macro_f1"] for d in SOURCES]))
    return results


def train_epoch(model: PACSClassifier, discriminator: DomainDiscriminator | None, optimizer: torch.optim.Optimizer, root: Path, protocol: dict, config: dict, epoch: int, device: torch.device, workers: int, steps: int) -> dict[str, float]:
    model.train()
    freeze_batchnorm_statistics(model)
    if discriminator is not None:
        discriminator.train()
    source_loaders = {
        domain: make_train_loader(
            root, protocol["source_splits"][domain]["train"], True, 8, steps,
            SEED + 100_000 * i + 10_000 * epoch, workers, device,
        )
        for i, domain in enumerate(SOURCES)
    }
    source_iters = {domain: iter(loader) for domain, loader in source_loaders.items()}
    target_iter = None
    if config["method"] != "source_only":
        target_loader = make_train_loader(
            root, protocol["target_unlabeled"], False, 24, steps,
            SEED + 900_000 + 10_000 * epoch, workers, device,
        )
        target_iter = iter(target_loader)
    totals = {"classification_loss": 0.0, "alignment_loss": 0.0, "domain_loss": 0.0, "domain_accuracy": 0.0}
    for step in range(steps):
        batches = [next(source_iters[domain]) for domain in SOURCES]
        source_images = torch.cat([batch[0] for batch in batches]).to(device, non_blocking=True)
        source_labels = torch.cat([batch[1] for batch in batches]).to(device, non_blocking=True)
        if len(source_labels) != 24:
            raise RuntimeError("A source update must contain eight examples from each domain")
        optimizer.zero_grad(set_to_none=True)
        if target_iter is None:
            source_logits, _ = model(source_images)
            features, logits = None, None
        else:
            target_images, _ = next(target_iter)
            if len(target_images) != 24:
                raise RuntimeError("An adaptation update must contain 24 target examples")
            logits, features = model(torch.cat((source_images, target_images.to(device, non_blocking=True))))
            source_logits = logits[:24]
        classification_loss = nn.functional.cross_entropy(source_logits, source_labels)
        loss = classification_loss
        alignment_loss = classification_loss.new_zeros(())
        domain_loss = classification_loss.new_zeros(())
        domain_accuracy = 0.0
        if config["method"] == "dan":
            alignment_loss = three_kernel_mmd(features[:24], features[24:])
            loss = loss + config["mmd_lambda"] * alignment_loss
        elif config["method"] in ("dann", "cdan"):
            progress = (epoch * steps + step) / max(1, config["epochs"] * steps - 1)
            strength = reversal_strength(progress, config["grl_maximum"])
            domain_input = conditional_features(features, logits) if config["method"] == "cdan" else features
            domain_logits = discriminator(reverse_gradient(domain_input, strength))
            domain_labels = torch.cat((torch.zeros(24, dtype=torch.long, device=device), torch.ones(24, dtype=torch.long, device=device)))
            domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
            domain_accuracy = float((domain_logits.argmax(1) == domain_labels).float().mean().item())
            loss = loss + domain_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step + 1}")
        loss.backward()
        optimizer.step()
        totals["classification_loss"] += float(classification_loss.detach())
        totals["alignment_loss"] += float(alignment_loss.detach())
        totals["domain_loss"] += float(domain_loss.detach())
        totals["domain_accuracy"] += domain_accuracy
    return {key: value / steps for key, value in totals.items()}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_code_sha256() -> str:
    repo = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in (
        "shared/pacs.py", "shared/mmd.py", "shared/make_pacs_protocol.py",
        "task2/model.py", "task2/methods.py", "task2/train.py",
    ):
        path = repo / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    config = load_config(args.run_id)
    root = resolve_root(args.pacs_root)
    protocol = load_protocol(args.protocol)
    verify_dataset_snapshot(root, protocol)
    output = args.output / args.run_id
    if args.resume:
        if not (output / "last.pt").exists():
            raise FileNotFoundError(f"No resumable epoch checkpoint in {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PACSClassifier(pretrained=True).to(device)
    initial_model_sha256 = hash_model(model)
    original_bn = batchnorm_buffers(model)
    discriminator = DomainDiscriminator(config["method"] == "cdan").to(device) if config["method"] in ("dann", "cdan") else None
    parameters = list(model.parameters()) + (list(discriminator.parameters()) if discriminator is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    validation = {
        domain: make_validation_loader(root, protocol["source_splits"][domain]["validation"], args.num_workers, device)
        for domain in SOURCES
    }
    steps = steps_per_source_epoch(protocol)
    history = []
    best_f1, best_epoch, stale = float("-inf"), None, 0
    start_epoch = 0
    if args.resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["file_list_sha256"] != protocol["file_list_sha256"] or saved["initial_model_sha256"] != initial_model_sha256:
            raise RuntimeError("Resume checkpoint does not match this experiment")
        model.load_state_dict(saved["model_state"])
        if discriminator is not None:
            discriminator.load_state_dict(saved["discriminator_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history = saved["history"]
        best_f1, best_epoch, stale = saved["best_f1"], saved["best_epoch"], saved["stale"]
        start_epoch = saved["completed_epochs"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        assert_batchnorm_unchanged(model, original_bn)
    for epoch in range(start_epoch, config["epochs"]):
        if stale >= config["patience"]:
            break
        train_metrics = train_epoch(model, discriminator, optimizer, root, protocol, config, epoch, device, args.num_workers, steps)
        assert_batchnorm_unchanged(model, original_bn)
        validation_metrics = validate_sources(model, validation, device)
        row = {"epoch": epoch + 1, **train_metrics, **validation_metrics}
        history.append(row)
        if validation_metrics["mean_source_macro_f1"] > best_f1 + 1e-12:
            best_f1, best_epoch, stale = validation_metrics["mean_source_macro_f1"], epoch + 1, 0
            temporary = output / "best.pt.tmp"
            torch.save({"model_state": model.state_dict(), "discriminator_state": discriminator.state_dict() if discriminator is not None else None, "epoch": best_epoch, "config": config, "file_list_sha256": protocol["file_list_sha256"]}, temporary)
            os.replace(temporary, output / "best.pt")
            (output / "best_source_validation.json").write_text(json.dumps({"epoch": best_epoch, **validation_metrics}, indent=2) + "\n")
        else:
            stale += 1
        with (output / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        temporary_last = output / "last.pt.tmp"
        torch.save({
            "config": config, "file_list_sha256": protocol["file_list_sha256"],
            "initial_model_sha256": initial_model_sha256,
            "model_state": model.state_dict(),
            "discriminator_state": discriminator.state_dict() if discriminator is not None else None,
            "optimizer_state": optimizer.state_dict(), "history": history,
            "best_f1": best_f1, "best_epoch": best_epoch, "stale": stale,
            "completed_epochs": epoch + 1,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, temporary_last)
        os.replace(temporary_last, output / "last.pt")
        print(json.dumps({"run": args.run_id, "epoch": epoch + 1, "best_epoch": best_epoch, "stale": stale, **train_metrics, **validation_metrics}), flush=True)
        if stale >= config["patience"]:
            break
    manifest = {
        "config": config, "protocol_file_list_sha256": protocol["file_list_sha256"],
        "dataset_source": args.dataset_source,
        "git_commit": git_commit(), "source_code_sha256": source_code_sha256(),
        "initial_model_sha256": initial_model_sha256,
        "best_epoch": best_epoch, "best_mean_source_macro_f1": best_f1,
        "epochs_completed": len(history), "steps_per_epoch": steps,
        "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "checkpoint_sha256": hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(),
        "target_labels_used": False,
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=tuple(MAIN_RUNS), required=True)
    parser.add_argument("--pacs-root", type=Path, required=True)
    parser.add_argument("--dataset-source", required=True, help="PACS archive URL, Kaggle dataset version, or other precise source")
    parser.add_argument("--protocol", type=Path, default=Path("shared/splits/pacs_sketch_seed6304.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/task2"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Resume at the next source epoch from last.pt")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
