"""Create a readable presentation copy of the saved separability scatter."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "task2/results/final/all_six_comparison.csv"
OUTPUT = ROOT / "task2/results/presentation/separability_vs_target_accuracy_clean.png"
RECORD = ROOT / "task2/results/presentation/separability_vs_target_accuracy_clean.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise RuntimeError("Expected all six frozen runs")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for row in rows:
        axis.scatter(
            float(row["domain_separability"]),
            float(row["target_accuracy"]),
            s=70,
            label=row["method"],
        )
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    axis.set_xlim(0.48, 1.01)
    axis.set_ylim(0.0, 0.80)
    axis.set_xlabel("Source-vs-Sketch separability accuracy")
    axis.set_ylabel("Sketch accuracy")
    axis.grid(alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)

    record = {
        "status": "presentation_only",
        "models_executed": False,
        "metrics_changed": False,
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(SOURCE),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(OUTPUT),
        "change": "Replace overlapping point labels with an external legend.",
        "matplotlib_version": matplotlib.__version__,
    }
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
