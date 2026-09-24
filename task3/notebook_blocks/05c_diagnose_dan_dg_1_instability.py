# TASK 3 — DAN-DG LAMBDA=1 INSTABILITY DIAGNOSTIC
#
# Reads existing CSV histories only.
# Does not load models, datasets, or Sketch.

from pathlib import Path
import csv
import json
import math
import os


STEPS_PER_EPOCH = 235

TASK3_ROOT = Path(
    "/content/drive/MyDrive/ATML-PA1/"
    "task3_domain_generalization_20260924"
)
TASK3_HISTORY = (
    TASK3_ROOT
    / "training"
    / "dan_dg_1"
    / "history.csv"
)
DIAGNOSTIC_PATH = (
    TASK3_ROOT
    / "provenance"
    / "dan_dg_1_instability_diagnostic.json"
)

CODE_ROOT = Path("/content/atml_pa1_task3_source")
TASK2_DAN_HISTORY = (
    CODE_ROOT
    / "task2"
    / "results"
    / "training"
    / "dan_1"
    / "history.csv"
)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_json(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    os.replace(temporary, path)


task3_rows = read_csv(TASK3_HISTORY)

if len(task3_rows) != 8:
    raise RuntimeError(
        f"Expected 8 Task 3 epochs, found {len(task3_rows)}"
    )

pairs = (
    "photo__art_painting",
    "photo__cartoon",
    "art_painting__cartoon",
)

diagnostic_rows = []

print("Task 3 DAN-DG lambda=1 detailed diagnostics:\n")

for row in task3_rows:
    epoch = int(row["epoch"])
    clipped_fraction = float(
        row["gradient_clipped_fraction"]
    )
    clipped_steps = round(
        clipped_fraction * STEPS_PER_EPOCH
    )

    pair_details = {}

    for pair in pairs:
        mmd = float(row[f"mmd_{pair}"])
        median = float(row[f"mmd_median_{pair}"])
        zeros = float(
            row[f"mmd_off_diagonal_zeros_{pair}"]
        )

        if not all(
            math.isfinite(value)
            for value in (mmd, median, zeros)
        ):
            raise RuntimeError(
                f"Non-finite pair diagnostic at epoch {epoch}"
            )

        if median <= 0:
            raise RuntimeError(
                f"Non-positive bandwidth median at epoch {epoch}"
            )

        pair_details[pair] = {
            "mmd": mmd,
            "median_squared_distance": median,
            "mean_off_diagonal_zero_count": zeros,
        }

    record = {
        "epoch": epoch,
        "classification_loss": float(
            row["classification_loss"]
        ),
        "mean_pairwise_mmd": float(
            row["mmd_loss"]
        ),
        "gradient_norm_before_clipping": float(
            row["gradient_norm"]
        ),
        "gradient_norm_after_clipping": float(
            row["gradient_norm_after_clipping"]
        ),
        "clipped_fraction": clipped_fraction,
        "estimated_clipped_steps": clipped_steps,
        "mean_source_macro_f1": float(
            row["mean_source_macro_f1"]
        ),
        "worst_source_macro_f1": float(
            row["worst_source_macro_f1"]
        ),
        "pairs": pair_details,
    }

    diagnostic_rows.append(record)

    medians = " ".join(
        f"{pair}={pair_details[pair]['median_squared_distance']:.8g}"
        for pair in pairs
    )

    mmd_values = " ".join(
        f"{pair}={pair_details[pair]['mmd']:.6f}"
        for pair in pairs
    )

    zero_values = " ".join(
        f"{pair}={pair_details[pair]['mean_off_diagonal_zero_count']:.3f}"
        for pair in pairs
    )

    print(
        f"\nEpoch {epoch:02d}\n"
        f"  classification: "
        f"{record['classification_loss']:.6f}\n"
        f"  mean MMD:       "
        f"{record['mean_pairwise_mmd']:.6f}\n"
        f"  gradient:       "
        f"{record['gradient_norm_before_clipping']:.6f}"
        f" -> "
        f"{record['gradient_norm_after_clipping']:.6f}\n"
        f"  clipped steps:  "
        f"{clipped_steps}/{STEPS_PER_EPOCH}\n"
        f"  mean/worst F1:  "
        f"{record['mean_source_macro_f1']:.6f} / "
        f"{record['worst_source_macro_f1']:.6f}\n"
        f"  medians:        {medians}\n"
        f"  pair MMDs:      {mmd_values}\n"
        f"  zero counts:    {zero_values}"
    )


# Compare against Task 2 DAN lambda=1 using source-side
# training diagnostics only. No Task 2 Sketch result is read.
task2_comparison = None

if TASK2_DAN_HISTORY.is_file():
    task2_rows = read_csv(TASK2_DAN_HISTORY)

    task2_comparison = [
        {
            "epoch": int(row["epoch"]),
            "mmd_loss": float(row["mmd_loss"]),
            "median_squared_distance": float(
                row["mmd_median_squared_distance"]
            ),
            "gradient_norm_before_clipping": float(
                row["gradient_norm"]
            ),
            "gradient_norm_after_clipping": float(
                row["gradient_norm_after_clipping"]
            ),
            "clipped_fraction": float(
                row["gradient_clipped_fraction"]
            ),
            "mean_source_macro_f1": float(
                row["mean_source_macro_f1"]
            ),
        }
        for row in task2_rows
    ]

    print(
        "\n\nTask 2 DAN lambda=1 source-side reference:"
    )

    for row in task2_comparison:
        print(
            f"epoch={row['epoch']:02d} "
            f"median="
            f"{row['median_squared_distance']:.8g} "
            f"gradient="
            f"{row['gradient_norm_before_clipping']:.3f}"
            f" -> "
            f"{row['gradient_norm_after_clipping']:.3f} "
            f"clip_fraction="
            f"{row['clipped_fraction']:.4f} "
            f"mean_f1="
            f"{row['mean_source_macro_f1']:.6f}"
        )
else:
    print(
        "\nTask 2 source-side DAN history was not found; "
        "the Task 3 diagnostic remains complete."
    )


initial_medians = {
    pair: diagnostic_rows[0]["pairs"][pair][
        "median_squared_distance"
    ]
    for pair in pairs
}

final_medians = {
    pair: diagnostic_rows[-1]["pairs"][pair][
        "median_squared_distance"
    ]
    for pair in pairs
}

median_ratios = {
    pair: final_medians[pair] / initial_medians[pair]
    for pair in pairs
}

diagnostic = {
    "status":
        "TASK3_DAN_DG_1_SOURCE_ONLY_DIAGNOSTIC_COMPLETE",
    "run_id": "dan_dg_1",
    "mmd_lambda": 1.0,
    "source_only_phase": True,
    "sketch_images_accessed": 0,
    "reason": (
        "Universal or near-universal gradient clipping, "
        "rising pre-clip norms, and unstable source "
        "validation performance."
    ),
    "task3_epochs": diagnostic_rows,
    "initial_pair_medians": initial_medians,
    "final_pair_medians": final_medians,
    "final_to_initial_median_ratios": median_ratios,
    "task2_dan_1_source_only_reference":
        task2_comparison,
}

if DIAGNOSTIC_PATH.exists():
    existing = json.loads(
        DIAGNOSTIC_PATH.read_text()
    )
    if existing != diagnostic:
        raise RuntimeError(
            "A different instability diagnostic already exists"
        )
else:
    atomic_write_json(
        diagnostic,
        DIAGNOSTIC_PATH,
    )

print("\n\nMedian final/initial ratios:")
print(json.dumps(median_ratios, indent=2))

print(
    f"\nSaved diagnostic: {DIAGNOSTIC_PATH}"
)
print("Sketch images accessed: 0")
print(
    "Do not begin another run until this diagnostic "
    "has been reviewed."
)
