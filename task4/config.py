from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_COMMON = {
    "run_id",
    "method",
    "seed",
    "model",
    "data",
    "training",
    "selection",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML file and validate the locked PA settings."""
    path = Path(path)
    config = json.loads(path.read_text())
    missing = REQUIRED_COMMON - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["method"] not in {"vanilla", "gcsc", "proser"}:
        raise ValueError("Task 4 permits only vanilla, gcsc, and proser")
    if config["seed"] != 6304:
        raise ValueError("The PA locks Task 4 to seed 6304")
    if config["model"] != {
        "architecture": "resnet18_cifar",
        "num_known_classes": 10,
    }:
        raise ValueError("The PA requires the same ten-class CIFAR ResNet-18")
    data = config["data"]
    if data.get("validation_fraction") != 0.1:
        raise ValueError("The PA requires a stratified 90/10 CIFAR-10 split")
    training = config["training"]
    expected_epochs = 50 if config["method"] == "proser" else 100
    expected_lr = 1e-3 if config["method"] == "proser" else 0.1
    locked = {
        "epochs": expected_epochs,
        "batch_size": 128,
        "learning_rate": expected_lr,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "schedule": "cosine",
    }
    for key, expected in locked.items():
        if training.get(key) != expected:
            raise ValueError(f"Locked setting {key!r} must equal {expected!r}")
    if config["selection"] != {"metric": "validation_accuracy", "mode": "max"}:
        raise ValueError("Checkpoints must be selected by CIFAR-10 validation accuracy")
    if config["method"] == "gcsc":
        if config.get("randaugment") != {"num_ops": 2, "magnitude": 9}:
            raise ValueError("GCSC requires RandAugment(num_ops=2, magnitude=9)")
    if config["method"] == "proser":
        expected = {
            "dummy_classifiers": 5,
            "beta": 1.0,
            "gamma": 0.1,
            "mixup_alpha": 2.0,
            "mixup_beta": 2.0,
            "mix_after": "layer2",
            "batch_halves": ["classifier_placeholder", "data_placeholder"],
        }
        if config.get("proser") != expected:
            raise ValueError("PROSER settings differ from the PA-locked protocol")


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

