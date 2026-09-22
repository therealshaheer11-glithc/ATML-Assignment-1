"""Interactively record preregistered AdaIN candidate reviews in Colab.

Run this file with IPython ``%run`` so widget callbacks remain in the notebook
kernel.  The reviewer sees the content image, generated output, and style image;
no classifier outputs are loaded or displayed.  Decisions are written
atomically to the persistent review manifest after every button press.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image
from torchvision.datasets import STL10


REJECTION_REASONS = (
    "content object is no longer visually recognizable",
    "object is severely obscured or structurally corrupted",
    "output contains blank regions or major rendering artifacts",
    "no perceptible style or texture transfer occurred",
)
REVIEW_FIELDS = (
    "candidate_id",
    "content_official_index",
    "content_class_name",
    "style_official_index",
    "style_class_name",
    "output_sha256",
    "review_status",
    "rejection_reason",
    "reviewed_by",
    "review_completed_before_model_evaluation",
)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if any(field not in fields for field in REVIEW_FIELDS):
        raise RuntimeError("Review manifest lacks required fields.")
    if not rows or len({row["candidate_id"] for row in rows}) != len(rows):
        raise RuntimeError("Review manifest is empty or contains duplicate IDs.")
    return fields, rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_pre_evaluation_state(root: Path) -> None:
    for directory in (
        root / "metrics/cue_conflict",
        root / "predictions/cue_conflict",
    ):
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                "Cue-conflict model outputs already exist; review decisions are "
                "frozen. Use analysis/browse_cue_examples.py for read-only inspection."
            )


def candidate_ids_for_scope(
    scope: str,
    manifest_dir: Path,
    rows: list[dict[str, str]],
) -> list[str]:
    if scope == "initial":
        if len(rows) != 200 or not all(
            row["part_of_initial_generation"] == "True" for row in rows
        ):
            raise RuntimeError("Initial review requires exactly 200 initial candidates.")
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row["pair_number"]),
                row["direction_code"],
                int(row["schedule_rank"]),
            ),
        )
        return [row["candidate_id"] for row in ordered]

    round_number = int(scope.removeprefix("round-"))
    plan_path = manifest_dir / f"replacement_plan_round_{round_number:02d}.json"
    snapshot_path = manifest_dir / f"review_snapshot_round_{round_number:02d}.csv"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _, snapshot = read_manifest(snapshot_path)
    ids = [item["candidate_id"] for item in plan["planned_candidates"]]
    if rows[: len(snapshot)] != snapshot:
        raise RuntimeError("Reviews preceding this replacement round changed.")
    if len(rows) != len(snapshot) + len(ids):
        raise RuntimeError("Replacement review manifest has an unexpected length.")
    if [row["candidate_id"] for row in rows[len(snapshot) :]] != ids:
        raise RuntimeError("Replacement candidates differ from the saved plan.")
    return ids


def atomic_write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def launch(
    artifacts_root: str | Path,
    reviewer: str,
    scope: str,
) -> None:
    root = Path(artifacts_root).resolve()
    if not reviewer.strip():
        raise ValueError("Reviewer name must not be empty.")
    require_pre_evaluation_state(root)

    manifest_dir = root / "cue_conflict/manifests"
    manifest_path = manifest_dir / "review_manifest.csv"
    candidate_dir = root / "cue_conflict/candidates"
    fields, initial_rows = read_manifest(manifest_path)
    candidate_ids = candidate_ids_for_scope(scope, manifest_dir, initial_rows)
    candidate_set = set(candidate_ids)
    if len(candidate_set) != len(candidate_ids):
        raise RuntimeError("Review scope contains duplicate candidate IDs.")

    for row in initial_rows:
        if row["candidate_id"] not in candidate_set:
            continue
        image_path = candidate_dir / f"{row['candidate_id']}.png"
        if not image_path.is_file() or sha256(image_path) != row["output_sha256"]:
            raise RuntimeError(f"Candidate image changed: {row['candidate_id']}")
        if row["review_status"] not in ("pending", "accepted", "rejected"):
            raise RuntimeError(f"Invalid review status: {row['candidate_id']}")
        if row["model_predictions_used_for_generation"] != "False":
            raise RuntimeError("A candidate records model-assisted generation.")
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                raise RuntimeError(f"Invalid candidate image: {row['candidate_id']}")

    dataset = STL10(
        root=str(root / "datasets/stl10"), split="test", download=False
    )
    if len(dataset) != 8000:
        raise RuntimeError("Official STL-10 test partition is unavailable.")

    def current_rows() -> tuple[list[str], list[dict[str, str]]]:
        current_fields, rows = read_manifest(manifest_path)
        if current_fields != fields:
            raise RuntimeError("Review-manifest columns changed during review.")
        if [row["candidate_id"] for row in rows] != [
            row["candidate_id"] for row in initial_rows
        ]:
            raise RuntimeError("Review-manifest row order changed during review.")
        return current_fields, rows

    by_id = {row["candidate_id"]: row for row in initial_rows}
    first_pending = next(
        (i for i, candidate_id in enumerate(candidate_ids)
         if by_id[candidate_id]["review_status"] == "pending"),
        0,
    )
    position = widgets.IntSlider(
        value=first_pending,
        min=0,
        max=len(candidate_ids) - 1,
        description="Candidate",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    heading = widgets.HTML()
    notice = widgets.HTML()
    view = widgets.Output()

    def show(*_: object) -> None:
        _, rows = current_rows()
        lookup = {row["candidate_id"]: row for row in rows}
        row = lookup[candidate_ids[position.value]]
        counts = Counter(lookup[candidate_id]["review_status"]
                         for candidate_id in candidate_ids)
        heading.value = (
            f"<b>{position.value + 1}/{len(candidate_ids)} - "
            f"{row['candidate_id']}</b><br>"
            f"{row['direction']} | Decision: {row['review_status']}<br>"
            f"Accepted: {counts['accepted']} | Rejected: {counts['rejected']} | "
            f"Pending: {counts['pending']}"
        )
        content, _ = dataset[int(row["content_official_index"])]
        style, _ = dataset[int(row["style_official_index"])]
        with Image.open(candidate_dir / f"{row['candidate_id']}.png") as image:
            generated = image.convert("RGB").copy()
        with view:
            clear_output(wait=True)
            figure, axes = plt.subplots(1, 3, figsize=(16, 5))
            for axis, image, title in zip(
                axes,
                (content, generated, style),
                (
                    f"C: shape - {row['content_class_name']}",
                    "O: REVIEW THIS OUTPUT",
                    f"S: texture - {row['style_class_name']}",
                ),
            ):
                axis.imshow(image)
                axis.set_title(title)
                axis.axis("off")
            figure.tight_layout()
            display(figure)
            plt.close(figure)

    def next_pending() -> None:
        _, rows = current_rows()
        lookup = {row["candidate_id"]: row for row in rows}
        for step in range(1, len(candidate_ids) + 1):
            index = (position.value + step) % len(candidate_ids)
            if lookup[candidate_ids[index]]["review_status"] == "pending":
                position.value = index
                return
        notice.value = "<b>Every candidate in this scope has been reviewed.</b>"
        show()

    def record(status: str, reason: str = "") -> None:
        current_fields, rows = current_rows()
        row = next(
            row for row in rows
            if row["candidate_id"] == candidate_ids[position.value]
        )
        if status != "pending" and row["review_status"] not in ("pending", status):
            notice.value = "Use Clear decision before changing this choice."
            return
        row["review_status"] = status
        row["rejection_reason"] = reason
        row["reviewed_by"] = reviewer.strip() if status != "pending" else ""
        row["review_completed_before_model_evaluation"] = (
            "True" if status != "pending" else ""
        )
        atomic_write(manifest_path, current_fields, rows)
        notice.value = ""
        show() if status == "pending" else next_pending()

    previous = widgets.Button(description="Previous")
    following = widgets.Button(description="Next")
    pending = widgets.Button(description="Next pending")
    clear = widgets.Button(description="Clear decision", button_style="warning")
    previous.on_click(
        lambda _: setattr(position, "value", (position.value - 1) % len(candidate_ids))
    )
    following.on_click(
        lambda _: setattr(position, "value", (position.value + 1) % len(candidate_ids))
    )
    pending.on_click(lambda _: next_pending())
    clear.on_click(lambda _: record("pending"))

    choices = [("Accept & next", "accepted", "", "success")]
    choices.extend(
        (f"Reject R{index} & next", "rejected", reason, "danger")
        for index, reason in enumerate(REJECTION_REASONS, 1)
    )
    buttons = []
    for label, status, reason, color in choices:
        button = widgets.Button(description=label, button_style=color)
        button.on_click(
            lambda _, selected_status=status, selected_reason=reason:
            record(selected_status, selected_reason)
        )
        buttons.append(button)

    position.observe(show, names="value")
    display(
        widgets.VBox([
            heading,
            position,
            view,
            notice,
            widgets.HBox([previous, following, pending, clear]),
            widgets.HBox(buttons),
        ])
    )
    show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--scope", choices=("initial", "round-1", "round-2"), required=True
    )
    args = parser.parse_args()
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except ImportError:
        pass
    launch(args.artifacts_root, args.reviewer, args.scope)


if __name__ == "__main__":
    main()
