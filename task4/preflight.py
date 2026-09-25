from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy
import sklearn
import torch
import torchvision

from task4.config import canonical_config_sha256, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    configs = {}
    for method in ("vanilla", "gcsc", "proser"):
        path = Path(__file__).parent / "configs" / f"{method}.yaml"
        config = load_config(path)
        configs[method] = canonical_config_sha256(config)
    record = {
        "status": "TASK4_PREFLIGHT_PASS",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "config_sha256": configs,
        "seed": 6304,
        "methods": ["vanilla", "gcsc", "proser"],
        "cifar100_images_accessed": 0,
    }
    rendered = json.dumps(record, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()

