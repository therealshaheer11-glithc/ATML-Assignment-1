"""Freeze all Task 4 score definitions and thresholds before CIFAR-100 access."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from task4.config import canonical_config_sha256, load_config
from task4.data.cifar10 import file_sha256, prepare_cifar10, unaugmented_training_subset
from task4.evaluation.metrics import validation_threshold
from task4.methods.proser import calibrate_dummy_bias, placeholder_unknownness
from task4.outputs import extract_outputs, load_model_from_checkpoint
from task4.runtime import atomic_json_dump, seed_everything
from task4.scores import (
    energy_unknownness,
    fit_diagonal_mahalanobis,
    mls_unknownness,
    msp_unknownness,
)


def tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--vanilla-config", required=True)
    parser.add_argument("--vanilla-checkpoint", required=True)
    parser.add_argument("--gcsc-config", required=True)
    parser.add_argument("--gcsc-checkpoint", required=True)
    parser.add_argument("--proser-config", required=True)
    parser.add_argument("--proser-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    lock_path = output / "evaluation_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"Evaluation lock already exists: {lock_path}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(6304)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_paths = {
        "vanilla": args.vanilla_config,
        "gcsc": args.gcsc_config,
        "proser": args.proser_config,
    }
    checkpoint_paths = {
        "vanilla": args.vanilla_checkpoint,
        "gcsc": args.gcsc_checkpoint,
        "proser": args.proser_checkpoint,
    }
    configs = {name: load_config(path) for name, path in config_paths.items()}
    for name, config in configs.items():
        if config["method"] != name:
            raise ValueError(f"{name} configuration has method {config['method']!r}")

    _, validation_set, _, _ = prepare_cifar10(
        args.data_root, args.split_manifest, "vanilla", args.download
    )
    training_set = unaugmented_training_subset(args.data_root, args.split_manifest)
    train_loader = DataLoader(training_set, batch_size=256, shuffle=False, num_workers=2)
    validation_loader = DataLoader(validation_set, batch_size=256, shuffle=False, num_workers=2)

    models = {}
    checkpoints = {}
    for name in ("vanilla", "gcsc", "proser"):
        model, checkpoint = load_model_from_checkpoint(checkpoint_paths[name], name, device)
        expected_config = canonical_config_sha256(configs[name])
        if checkpoint.get("config_sha256") != expected_config:
            raise ValueError(f"{name} checkpoint does not match its configuration")
        if checkpoint.get("split_manifest_sha256") != file_sha256(args.split_manifest):
            raise ValueError(f"{name} checkpoint does not match the split manifest")
        models[name] = model
        checkpoints[name] = checkpoint

    vanilla_train = extract_outputs(models["vanilla"], train_loader, device)
    validation = {
        name: extract_outputs(model, validation_loader, device) for name, model in models.items()
    }
    mahalanobis = fit_diagonal_mahalanobis(
        tensor(vanilla_train.features), tensor(vanilla_train.labels), diagonal_epsilon=1e-6
    )
    np.savez_compressed(
        output / "mahalanobis_stats.npz",
        class_means=mahalanobis.class_means.numpy(),
        shared_variance=mahalanobis.shared_variance.numpy(),
    )

    vanilla_logits = tensor(validation["vanilla"].known_logits)
    vanilla_features = tensor(validation["vanilla"].features)
    vanilla_scores = {
        "msp": msp_unknownness(vanilla_logits).numpy(),
        "mls": mls_unknownness(vanilla_logits).numpy(),
        "energy": energy_unknownness(vanilla_logits).numpy(),
        "mahalanobis": mahalanobis.unknownness(vanilla_features).numpy(),
    }
    thresholds = {
        f"vanilla:{score_name}": validation_threshold(scores)
        for score_name, scores in vanilla_scores.items()
    }
    thresholds["gcsc:mls"] = validation_threshold(
        mls_unknownness(tensor(validation["gcsc"].known_logits)).numpy()
    )
    thresholds["proser:mls"] = validation_threshold(
        mls_unknownness(tensor(validation["proser"].known_logits)).numpy()
    )
    proser_dummy = validation["proser"].dummy_logits
    if proser_dummy is None:
        raise RuntimeError("PROSER validation outputs lack dummy logits")
    dummy_bias = calibrate_dummy_bias(
        tensor(validation["proser"].known_logits), tensor(proser_dummy)
    )
    proser_placeholder_validation = placeholder_unknownness(
        tensor(validation["proser"].known_logits), tensor(proser_dummy), dummy_bias
    ).numpy()
    thresholds["proser:placeholder"] = validation_threshold(proser_placeholder_validation)

    for name, outputs in validation.items():
        payload = {
            "known_logits": outputs.known_logits,
            "features": outputs.features,
            "labels": outputs.labels,
        }
        if outputs.dummy_logits is not None:
            payload["dummy_logits"] = outputs.dummy_logits
        np.savez_compressed(output / f"{name}_validation_outputs.npz", **payload)

    lock = {
        "status": "TASK4_EVALUATION_LOCKED",
        "seed": 6304,
        "known_dataset": "CIFAR-10",
        "unknown_dataset": "CIFAR-100 test only",
        "unknown_groups": {
            "near": ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"],
            "far": ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"],
        },
        "checkpoint_sha256": {
            name: file_sha256(path) for name, path in checkpoint_paths.items()
        },
        "config_sha256": {
            name: canonical_config_sha256(config) for name, config in configs.items()
        },
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "score_definitions": {
            "msp": "1 - max softmax probability over ten known logits",
            "mls": "negative maximum of ten known logits",
            "energy": "negative logsumexp over ten known logits",
            "mahalanobis": "minimum shared-diagonal class distance; variance epsilon 1e-6",
            "proser_placeholder": "reference delta probability: strongest calibrated dummy minus strongest known; temperature 1024",
        },
        "threshold_rule": "95th percentile of CIFAR-10 validation unknownness; NumPy linear quantile",
        "thresholds": thresholds,
        "proser_dummy_bias": dummy_bias,
        "proser_temperature": 1024.0,
        "mahalanobis_stats_sha256": file_sha256(output / "mahalanobis_stats.npz"),
        "cifar100_images_accessed": 0,
        "rpl_included": False,
    }
    atomic_json_dump(lock, lock_path)
    print(f"Task 4 evaluation locked without CIFAR-100 access: {lock_path}")


if __name__ == "__main__":
    main()
