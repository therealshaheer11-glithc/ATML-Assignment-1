"""Build Task 2 training-curve and alignment-strength evidence from saved results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def run(freeze_path: Path, final_dir: Path, output: Path) -> None:
    lock = json.loads(freeze_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    for run_id, record in lock["runs"].items():
        history = rows(Path(record["directory"]) / "history.csv")
        epochs = [int(row["epoch"]) for row in history]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
        axes[0].plot(epochs, [float(row["classification_loss"]) for row in history])
        axes[0].set_title("Source classification loss")
        if record["config"]["method"] == "dan":
            axes[1].plot(epochs, [float(row["alignment_loss"]) for row in history])
            axes[1].set_title("MMD penalty (unweighted)")
        elif record["config"]["method"] in ("dann", "cdan"):
            axes[1].plot(epochs, [float(row["domain_loss"]) for row in history], label="Domain loss")
            axes[1].plot(epochs, [float(row["domain_accuracy"]) for row in history], label="Domain accuracy")
            axes[1].set_title("Domain discriminator")
            axes[1].legend(fontsize=8)
        else:
            axes[1].axis("off")
        for domain in ("photo", "art_painting", "cartoon"):
            axes[2].plot(epochs, [float(row[f"{domain}_macro_f1"]) for row in history], label=domain)
        axes[2].plot(epochs, [float(row["mean_source_macro_f1"]) for row in history], label="mean", color="black", linewidth=2)
        axes[2].axvline(record["best_epoch"], color="gray", linestyle="--", linewidth=1)
        axes[2].set_title("Source validation macro-F1")
        axes[2].legend(fontsize=7)
        for axis in axes:
            if axis.axison:
                axis.set_xlabel("Source epoch")
                axis.grid(alpha=0.2)
        fig.suptitle(run_id)
        fig.tight_layout()
        fig.savefig(output / f"{run_id}_training.png", dpi=180)
        plt.close(fig)
    comparison = {row["run_id"]: row for row in rows(final_dir / "comparison.csv")}
    strengths = ("dan_0p1", "dan_1", "dan_10")
    lambdas = [0.1, 1.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for axis, field, title in zip(
        axes,
        ("mean_source_macro_f1", "domain_separability", "target_accuracy"),
        ("Mean source macro-F1", "Domain separability", "Sketch accuracy"),
    ):
        axis.plot(lambdas, [float(comparison[run][field]) for run in strengths], marker="o")
        axis.set_xscale("log")
        axis.set_xticks(lambdas, [str(value) for value in lambdas])
        axis.set_xlabel("DAN MMD weight λ")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "dan_strength_study.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.freeze, args.final_dir, args.output)


if __name__ == "__main__":
    main()
