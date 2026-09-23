"""Create the one common ImageNet-V1 ResNet/head initialization used by every run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torchvision

from task2.model import PACSClassifier, state_dict_sha256


SEED = 6304


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seed_everything()
    model = PACSClassifier(pretrained=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state_hash = state_dict_sha256(state)
    payload = {
        "state_dict": state,
        "state_dict_sha256": state_hash,
        "seed": SEED,
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }

    if args.output.exists():
        existing = torch.load(args.output, map_location="cpu", weights_only=False)
        if existing.get("state_dict_sha256") != state_hash:
            raise RuntimeError("Existing common initialization differs; refusing overwrite")
        print(json.dumps({"status": "already_verified", **{k: v for k, v in payload.items() if k != "state_dict"}}, indent=2))
        return

    atomic_torch_save(payload, args.output)
    print(json.dumps({"status": "created", **{k: v for k, v in payload.items() if k != "state_dict"}}, indent=2))


if __name__ == "__main__":
    main()

