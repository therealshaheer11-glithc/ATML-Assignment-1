"""Read-only protocol and dataset checks before any model training."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision

from shared.pacs import load_protocol, steps_per_source_epoch, verify_dataset_snapshot
from task2.config import RUNS, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacs-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("shared/splits/pacs_sketch_seed6304.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    dataset = verify_dataset_snapshot(args.pacs_root, protocol)
    configs = {run_id: load_config(run_id) for run_id in RUNS}
    steps = steps_per_source_epoch(protocol)
    summary = {
        "status": "PREFLIGHT_PASS",
        "training_started": False,
        "target_labels_accessed": False,
        "dataset": dataset,
        "source_counts": {
            domain: {
                split: len(protocol["source_splits"][domain][split])
                for split in ("train", "validation")
            }
            for domain in protocol["sources"]
        },
        "target_unlabeled_count": len(protocol["target_unlabeled"]),
        "steps_per_source_epoch": steps,
        "planned_updates": steps * 30,
        "registered_runs": list(configs),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

