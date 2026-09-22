"""Create the assignment's shared Task 2/3 PACS split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.pacs import make_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("shared/splits/pacs_sketch_seed6304.json"))
    args = parser.parse_args()
    protocol = make_protocol(args.pacs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Saved {args.output}; dataset file-list SHA256={protocol['file_list_sha256']}")
    for domain, splits in protocol["source_splits"].items():
        print(f"{domain}: train={len(splits['train'])}, validation={len(splits['validation'])}")
    print(f"sketch unlabeled={len(protocol['target_unlabeled'])}")


if __name__ == "__main__":
    main()
