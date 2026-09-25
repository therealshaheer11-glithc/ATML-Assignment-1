"""Run the one-time fixed Task 4 known/unknown evaluation after lock creation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from task4.data.cifar10 import CIFAR10_CLASSES, file_sha256, prepare_cifar10
from task4.evaluation.metrics import known_unknown_metrics
from task4.methods.proser import placeholder_unknownness
from task4.outputs import ModelOutputs, extract_outputs, load_model_from_checkpoint
from task4.runtime import atomic_json_dump
from task4.scores import DiagonalMahalanobis, energy_unknownness, mls_unknownness, msp_unknownness


def tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array)


def score_set(outputs: ModelOutputs, mahalanobis: DiagonalMahalanobis | None = None):
    logits = tensor(outputs.known_logits)
    scores = {
        "msp": msp_unknownness(logits).numpy(),
        "mls": mls_unknownness(logits).numpy(),
        "energy": energy_unknownness(logits).numpy(),
    }
    if mahalanobis is not None:
        scores["mahalanobis"] = mahalanobis.unknownness(tensor(outputs.features)).numpy()
    return scores


def comparisons(known, near, far, threshold: float):
    return {
        "near": known_unknown_metrics(known, near, threshold),
        "far": known_unknown_metrics(known, far, threshold),
        "all": known_unknown_metrics(known, np.concatenate([near, far]), threshold),
    }


def save_npz(path: Path, outputs: ModelOutputs) -> None:
    payload = {
        "known_logits": outputs.known_logits,
        "features": outputs.features,
        "labels": outputs.labels,
        "original_indices": outputs.original_indices,
    }
    if outputs.dummy_logits is not None:
        payload["dummy_logits"] = outputs.dummy_logits
    np.savez_compressed(path, **payload)


def write_metric_tables(output: Path, table_one: dict, table_two: dict) -> None:
    group_fields = [
        f"{group}_{metric}"
        for group in ("near", "far", "all")
        for metric in ("auroc", "known_acceptance_rate", "unknown_rejection_rate", "fpr_at_95_tpr", "threshold")
    ]
    with (output / "vanilla_score_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["score", *group_fields])
        writer.writeheader()
        for score, group_metrics in table_one.items():
            row = {"score": score}
            for group, metrics in group_metrics.items():
                row.update({f"{group}_{name}": value for name, value in metrics.items()})
            writer.writerow(row)
    with (output / "trained_model_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model_and_score", "closed_set_accuracy", *group_fields]
        )
        writer.writeheader()
        for name, model_metrics in table_two.items():
            row = {
                "model_and_score": name,
                "closed_set_accuracy": model_metrics["closed_set_accuracy"],
            }
            for group in ("near", "far", "all"):
                row.update(
                    {
                        f"{group}_{metric}": value
                        for metric, value in model_metrics[group].items()
                    }
                )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--evaluation-lock", required=True)
    parser.add_argument("--vanilla-checkpoint", required=True)
    parser.add_argument("--gcsc-checkpoint", required=True)
    parser.add_argument("--proser-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    lock_path = Path(args.evaluation_lock)
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "TASK4_EVALUATION_LOCKED" or lock.get("cifar100_images_accessed") != 0:
        raise ValueError("A valid target-free evaluation lock is required")
    checkpoint_paths = {
        "vanilla": args.vanilla_checkpoint,
        "gcsc": args.gcsc_checkpoint,
        "proser": args.proser_checkpoint,
    }
    for name, path in checkpoint_paths.items():
        if file_sha256(path) != lock["checkpoint_sha256"][name]:
            raise ValueError(f"{name} checkpoint changed after evaluation lock")
    if file_sha256(args.split_manifest) != lock["split_manifest_sha256"]:
        raise ValueError("CIFAR-10 split manifest changed after evaluation lock")
    stats_path = lock_path.parent / "mahalanobis_stats.npz"
    if file_sha256(stats_path) != lock["mahalanobis_stats_sha256"]:
        raise ValueError("Mahalanobis statistics changed after evaluation lock")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Final Task 4 output already exists: {output}")
    output.mkdir(parents=True)
    cache_dir = output / "outputs"
    cache_dir.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, known_test, _ = prepare_cifar10(
        args.data_root, args.split_manifest, "vanilla", args.download
    )
    known_loader = DataLoader(known_test, batch_size=256, shuffle=False, num_workers=2)

    # Deliberately import and construct real unknowns only after every lock gate passes.
    from task4.data.cifar100_unknowns import load_fixed_unknowns

    near_set, far_set = load_fixed_unknowns(args.data_root, args.download)
    near_loader = DataLoader(near_set, batch_size=256, shuffle=False, num_workers=2)
    far_loader = DataLoader(far_set, batch_size=256, shuffle=False, num_workers=2)

    model_outputs: dict[str, dict[str, ModelOutputs]] = {}
    for name, checkpoint_path in checkpoint_paths.items():
        model, _ = load_model_from_checkpoint(checkpoint_path, name, device)
        model_outputs[name] = {
            "known": extract_outputs(model, known_loader, device),
            "near": extract_outputs(model, near_loader, device),
            "far": extract_outputs(model, far_loader, device),
        }
        for group, outputs in model_outputs[name].items():
            save_npz(cache_dir / f"{name}_{group}.npz", outputs)

    stats = np.load(stats_path)
    mahalanobis = DiagonalMahalanobis(
        tensor(stats["class_means"]), tensor(stats["shared_variance"])
    )
    vanilla_scores = {
        group: score_set(outputs, mahalanobis)
        for group, outputs in model_outputs["vanilla"].items()
    }
    table_one = {}
    for score_name in ("msp", "mls", "energy", "mahalanobis"):
        threshold = lock["thresholds"][f"vanilla:{score_name}"]
        table_one[score_name] = comparisons(
            vanilla_scores["known"][score_name],
            vanilla_scores["near"][score_name],
            vanilla_scores["far"][score_name],
            threshold,
        )

    table_two = {}
    for name in ("vanilla", "gcsc", "proser"):
        outputs = model_outputs[name]
        known_mls = mls_unknownness(tensor(outputs["known"].known_logits)).numpy()
        near_mls = mls_unknownness(tensor(outputs["near"].known_logits)).numpy()
        far_mls = mls_unknownness(tensor(outputs["far"].known_logits)).numpy()
        table_two[f"{name}:mls"] = {
            "closed_set_accuracy": float(
                np.mean(outputs["known"].known_logits.argmax(1) == outputs["known"].labels)
            ),
            **comparisons(
                known_mls,
                near_mls,
                far_mls,
                lock["thresholds"][f"{name}:mls"],
            ),
        }

    proser = model_outputs["proser"]
    placeholder_scores = {}
    for group, outputs in proser.items():
        if outputs.dummy_logits is None:
            raise RuntimeError("PROSER outputs lack dummy logits")
        placeholder_scores[group] = placeholder_unknownness(
            tensor(outputs.known_logits),
            tensor(outputs.dummy_logits),
            lock["proser_dummy_bias"],
            lock["proser_temperature"],
        ).numpy()
    table_two["proser:placeholder"] = {
        "closed_set_accuracy": table_two["proser:mls"]["closed_set_accuracy"],
        **comparisons(
            placeholder_scores["known"],
            placeholder_scores["near"],
            placeholder_scores["far"],
            lock["thresholds"]["proser:placeholder"],
        ),
    }

    cifar100_classes = near_set.base.classes
    threshold = lock["thresholds"]["vanilla:mls"]
    failure_rows = []
    for group in ("near", "far"):
        outputs = model_outputs["vanilla"][group]
        scores = vanilla_scores[group]["mls"]
        accepted = np.flatnonzero(scores <= threshold)
        if len(accepted) < 3:
            raise RuntimeError(f"Fewer than three accepted {group} unknowns for failure analysis")
        selected = accepted[np.argsort(scores[accepted])[:3]]
        for position in selected:
            failure_rows.append(
                {
                    "group": group,
                    "cifar100_test_index": int(outputs.original_indices[position]),
                    "unknown_class": cifar100_classes[int(outputs.labels[position])],
                    "predicted_known_class": CIFAR10_CLASSES[int(outputs.known_logits[position].argmax())],
                    "mls_unknownness": float(scores[position]),
                    "threshold": float(threshold),
                }
            )
    with (output / "vanilla_mls_failures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failure_rows[0]))
        writer.writeheader()
        writer.writerows(failure_rows)

    write_metric_tables(output, table_one, table_two)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for axis, score_name in zip(axes, ("msp", "mls", "mahalanobis")):
        axis.hist(vanilla_scores["known"][score_name], bins=40, density=True, alpha=0.55, label="Known")
        axis.hist(vanilla_scores["near"][score_name], bins=40, density=True, alpha=0.55, label="Near")
        axis.hist(vanilla_scores["far"][score_name], bins=40, density=True, alpha=0.55, label="Far")
        axis.axvline(lock["thresholds"][f"vanilla:{score_name}"], color="black", linestyle="--")
        axis.set_title(score_name.upper())
        axis.set_xlabel("Unknownness")
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False)
    figure.savefig(output / "vanilla_score_distributions.png", dpi=200)
    plt.close(figure)

    results = {
        "status": "TASK4_FINAL_EVALUATION_COMPLETE",
        "evaluation_lock_sha256": file_sha256(lock_path),
        "known_test_count": len(known_test),
        "near_unknown_count": len(near_set),
        "far_unknown_count": len(far_set),
        "table_1_vanilla_scores": table_one,
        "table_2_trained_models": table_two,
        "failure_examples": failure_rows,
        "rpl_included": False,
    }
    atomic_json_dump(results, output / "final_results.json")
    atomic_json_dump(
        {
            "status": "TASK4_FINAL_EVALUATION_COMPLETE",
            "final_results_sha256": file_sha256(output / "final_results.json"),
            "failure_analysis_sha256": file_sha256(output / "vanilla_mls_failures.csv"),
            "vanilla_table_sha256": file_sha256(output / "vanilla_score_comparison.csv"),
            "trained_model_table_sha256": file_sha256(output / "trained_model_comparison.csv"),
            "score_figure_sha256": file_sha256(output / "vanilla_score_distributions.png"),
            "unknown_images_accessed": len(near_set) + len(far_set),
        },
        output / "completion.json",
    )
    print(f"Task 4 final evaluation complete: {output}")


if __name__ == "__main__":
    main()
