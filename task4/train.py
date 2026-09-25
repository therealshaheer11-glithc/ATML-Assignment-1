from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from task4.config import canonical_config_sha256, load_config
from task4.data.cifar10 import file_sha256, prepare_cifar10
from task4.methods.proser import proser_training_loss
from task4.models import CifarResNet18, ProserResNet18
from task4.runtime import atomic_json_dump, atomic_torch_save, seed_everything, seed_worker, write_history


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, drop_last: bool = False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=True,
    )


def validation_accuracy(model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(images)
            logits = output[0] if isinstance(model, ProserResNet18) else output
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
    return correct / total


def train_epoch(model, loader, optimizer, method: str, device: torch.device):
    model.train()
    totals = {"loss": 0.0, "known_ce": 0.0, "classifier_placeholder": 0.0, "data_placeholder": 0.0}
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if method == "proser":
            losses = proser_training_loss(model, images, labels, beta=1.0, gamma=0.1)
            loss = losses.total
            batch_metrics = {
                "known_ce": float(losses.known_classification.detach()),
                "classifier_placeholder": float(losses.classifier_placeholder.detach()),
                "data_placeholder": float(losses.data_placeholder.detach()),
            }
        else:
            loss = F.cross_entropy(model(images), labels)
            batch_metrics = {"known_ce": float(loss.detach()), "classifier_placeholder": 0.0, "data_placeholder": 0.0}
        loss.backward()
        optimizer.step()
        batch_size = len(labels)
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in batch_metrics.items():
            totals[name] += value * batch_size
        examples += batch_size
    return {name: value / examples for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vanilla-checkpoint")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    method = config["method"]
    if method == "proser" and not args.vanilla_checkpoint:
        parser.error("PROSER requires --vanilla-checkpoint")
    if method != "proser" and args.vanilla_checkpoint:
        parser.error("--vanilla-checkpoint is valid only for PROSER")

    output = Path(args.output)
    if (output / "run.json").exists():
        raise FileExistsError(f"Completed run already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    train_set, validation_set, _, _ = prepare_cifar10(
        args.data_root, args.split_manifest, method, args.download
    )
    batch_size = config["training"]["batch_size"]
    train_loader = make_loader(
        train_set, batch_size, True, config["seed"], drop_last=(method == "proser")
    )
    validation_loader = make_loader(validation_set, 256, False, config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vanilla_sha = None
    if method == "proser":
        vanilla_checkpoint = torch.load(args.vanilla_checkpoint, map_location="cpu", weights_only=False)
        if vanilla_checkpoint.get("method") != "vanilla":
            raise ValueError("PROSER must initialize from the selected Vanilla checkpoint")
        known_model = CifarResNet18()
        known_model.load_state_dict(vanilla_checkpoint["model_state"])
        model = ProserResNet18(known_model, dummy_classifiers=5)
        vanilla_sha = file_sha256(args.vanilla_checkpoint)
    else:
        model = CifarResNet18()
    model.to(device)

    settings = config["training"]
    optimizer = SGD(
        model.parameters(),
        lr=settings["learning_rate"],
        momentum=settings["momentum"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    config_sha = canonical_config_sha256(config)
    split_sha = file_sha256(args.split_manifest)
    best_accuracy = -1.0
    best_epoch = None
    history = []
    started = time.time()

    for epoch in range(1, settings["epochs"] + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, method, device)
        val_accuracy = validation_accuracy(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            "validation_accuracy": val_accuracy,
        }
        history.append(row)
        write_history(output / "history.csv", history)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            atomic_torch_save(
                {
                    "method": method,
                    "epoch": epoch,
                    "validation_accuracy": val_accuracy,
                    "model_state": model.state_dict(),
                    "config_sha256": config_sha,
                    "split_manifest_sha256": split_sha,
                    "vanilla_checkpoint_sha256": vanilla_sha,
                },
                output / "best.pt",
            )
        scheduler.step()
        print(f"{method} epoch {epoch:03d}: val_accuracy={val_accuracy:.6f}")

    atomic_json_dump(
        {
            "status": "TASK4_TRAINING_COMPLETE",
            "method": method,
            "seed": config["seed"],
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "best_checkpoint_sha256": file_sha256(output / "best.pt"),
            "config_sha256": config_sha,
            "split_manifest_sha256": split_sha,
            "vanilla_checkpoint_sha256": vanilla_sha,
            "elapsed_seconds": time.time() - started,
            "cifar100_images_accessed": 0,
        },
        output / "run.json",
    )


if __name__ == "__main__":
    main()

