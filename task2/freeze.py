"""Lock all six source-selected runs before Sketch labels are evaluated."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from task2.train import MAIN_RUNS, load_config, source_code_sha256


TRAINING_SOURCE_FILES = (
    "shared/pacs.py", "shared/mmd.py", "shared/make_pacs_protocol.py",
    "task2/model.py", "task2/methods.py", "task2/train.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_source_code_sha256(path: Path) -> str:
    """Match the training code fingerprint against the uploaded ZIP."""
    digest = hashlib.sha256()
    with ZipFile(path) as archive:
        for relative in TRAINING_SOURCE_FILES:
            digest.update(relative.encode())
            digest.update(archive.read(relative))
    return digest.hexdigest()


def freeze(runs: dict[str, Path], output: Path, code_archive: Path | None = None) -> dict:
    if set(runs) != set(MAIN_RUNS):
        raise ValueError(f"Expected exactly these six runs: {sorted(MAIN_RUNS)}")
    records = {}
    initial_hashes, protocols, commits, code_hashes, dataset_sources = set(), set(), set(), set(), set()
    for run_id, directory in sorted(runs.items()):
        meta = json.loads((directory / "run.json").read_text())
        checkpoint = directory / "best.pt"
        if meta["config"] != load_config(run_id) or meta["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError(f"Config or checkpoint changed for {run_id}")
        if meta["target_labels_used"] is not False:
            raise RuntimeError(f"Target-label boundary failed for {run_id}")
        if not 1 <= meta["best_epoch"] <= 30:
            raise RuntimeError(f"No valid source-selected checkpoint for {run_id}")
        with (directory / "history.csv").open(newline="") as file:
            history = list(csv.DictReader(file))
        epochs = meta["epochs_completed"]
        if not 1 <= epochs <= 30 or len(history) != epochs or meta["best_epoch"] > epochs:
            raise RuntimeError(f"Incomplete training history for {run_id}")
        if [int(row["epoch"]) for row in history] != list(range(1, epochs + 1)):
            raise RuntimeError(f"Nonconsecutive epochs for {run_id}")
        if epochs < 30 and epochs - meta["best_epoch"] < 5:
            raise RuntimeError(f"Training stopped before patience was met for {run_id}")
        best_f1 = max(float(row["mean_source_macro_f1"]) for row in history)
        if abs(best_f1 - meta["best_mean_source_macro_f1"]) > 1e-10:
            raise RuntimeError(f"Best source-validation score changed for {run_id}")
        initial_hashes.add(meta["initial_model_sha256"])
        protocols.add(meta["protocol_file_list_sha256"])
        commits.add(meta["git_commit"])
        code_hashes.add(meta["source_code_sha256"])
        dataset_sources.add(meta["dataset_source"])
        records[run_id] = {
            "directory": str(directory.resolve()),
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "best_epoch": meta["best_epoch"],
            "best_mean_source_macro_f1": meta["best_mean_source_macro_f1"],
            "config": meta["config"],
        }
    if len(initial_hashes) != 1 or len(protocols) != 1 or len(dataset_sources) != 1:
        raise RuntimeError("Runs do not share their initialization and PACS snapshot")
    if len(commits) != 1:
        raise RuntimeError("Runs do not share one recorded Git commit value")
    if len(code_hashes) != 1 or next(iter(code_hashes)) != source_code_sha256():
        raise RuntimeError("Runs do not match the current training source code")
    archive_hash = None
    if code_archive is not None:
        if archive_source_code_sha256(code_archive) != next(iter(code_hashes)):
            raise RuntimeError("Uploaded ZIP differs from the training source code")
        archive_hash = sha256(code_archive)
    elif None in commits:
        raise RuntimeError("ZIP-based runs require the uploaded code archive")
    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Task 2 final Sketch evaluation; no target-label metric was used for model selection",
        "initial_model_sha256": next(iter(initial_hashes)),
        "protocol_file_list_sha256": next(iter(protocols)),
        "dataset_source": next(iter(dataset_sources)),
        "git_commit": next(iter(commits)),
        "code_archive_sha256": archive_hash,
        "source_code_sha256": next(iter(code_hashes)),
        "runs": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Will not overwrite an existing freeze manifest: {output}")
    output.write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="run_id=/path/to/output/directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-archive", type=Path, help="Original uploaded code ZIP, required when runs have no Git commit")
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        name, path = item.split("=", 1)
        if name in runs:
            raise ValueError(f"Duplicate run: {name}")
        runs[name] = Path(path)
    lock = freeze(runs, args.output, args.code_archive)
    print(f"Frozen {len(lock['runs'])} runs at {args.output}")


if __name__ == "__main__":
    main()
