"""Pin, download, and record the approved AdaIN implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


REPOSITORY_URL = "https://github.com/naoto0804/pytorch-AdaIN.git"
RELEASE_TAG = "v0.0.0"
EXPECTED_COMMIT_PREFIX = "324eede"
PAPER_URL = (
    "https://openaccess.thecvf.com/content_iccv_2017/html/"
    "Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html"
)
ORIGINAL_REPOSITORY_URL = "https://github.com/xunhuang1995/AdaIN-style"
PYTORCH_REPOSITORY_URL = "https://github.com/naoto0804/pytorch-AdaIN"
RELEASE_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0"
)
LICENSE_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/blob/v0.0.0/LICENSE"
)
WEIGHT_URLS = {
    "decoder.pth": (
        "https://github.com/naoto0804/pytorch-AdaIN/releases/"
        "download/v0.0.0/decoder.pth"
    ),
    "vgg_normalised.pth": (
        "https://github.com/naoto0804/pytorch-AdaIN/releases/"
        "download/v0.0.0/vgg_normalised.pth"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(arguments: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def record_source_in_configuration(configuration_path: Path) -> None:
    text = configuration_path.read_text(encoding="utf-8")
    marker = "    implementation_source: null\n"
    already_recorded = (
        "executable_repository: "
        '"https://github.com/naoto0804/pytorch-AdaIN"'
    )

    if marker in text:
        replacement = """    implementation_source:
      method_paper: "https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html"
      original_authors_repository: "https://github.com/xunhuang1995/AdaIN-style"
      executable_repository: "https://github.com/naoto0804/pytorch-AdaIN"
      executable_implementation_status: "unofficial PyTorch reproduction"
      release: "v0.0.0"
      commit: "324eede"
      license: "MIT"
      weights_manifest: "task1/results/cue_conflict/adain_source.json"
      approved_by_user_before_generation: true
"""
        configuration_path.write_text(
            text.replace(marker, replacement, 1),
            encoding="utf-8",
        )
        print("PASS: AdaIN source recorded in task1.yaml.")
    elif already_recorded in text:
        print("PASS: AdaIN source was already recorded in task1.yaml.")
    else:
        raise RuntimeError(
            "Could not safely locate the AdaIN source field in task1.yaml."
        )


def prepare_external_repository(external_repository: Path) -> str:
    if external_repository.exists():
        if not (external_repository / ".git").is_dir():
            raise RuntimeError(
                f"Existing path is not the expected Git repository: "
                f"{external_repository}"
            )
    else:
        external_repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                RELEASE_TAG,
                "--depth",
                "1",
                REPOSITORY_URL,
                str(external_repository),
            ],
            check=True,
        )

    commit = run_git(["rev-parse", "HEAD"], cwd=external_repository)
    tag = run_git(
        ["describe", "--tags", "--exact-match"],
        cwd=external_repository,
    )

    if tag != RELEASE_TAG:
        raise RuntimeError(f"Expected tag {RELEASE_TAG}; observed {tag}.")
    if not commit.startswith(EXPECTED_COMMIT_PREFIX):
        raise RuntimeError(
            f"Expected commit beginning {EXPECTED_COMMIT_PREFIX}; "
            f"observed {commit}."
        )

    print(f"PASS: pinned AdaIN source {tag} at {commit}.")
    return commit


def download_file(url: str, destination: Path) -> None:
    if destination.is_file():
        print(f"Reusing existing file: {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ATML-PA1-reproducible-setup"},
    )

    downloaded = 0
    with urllib.request.urlopen(request) as response:
        if response.status not in (200, 206):
            raise RuntimeError(
                f"Download failed with HTTP status {response.status}: {url}"
            )

        with temporary_path.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if downloaded % (25 * 1024 * 1024) < len(block):
                    print(
                        f"{destination.name}: "
                        f"{downloaded / (1024 * 1024):.1f} MiB"
                    )

    if downloaded == 0:
        raise RuntimeError(f"Downloaded an empty file from {url}")

    os.replace(temporary_path, destination)
    print(
        f"PASS: downloaded {destination.name} "
        f"({downloaded / (1024 * 1024):.1f} MiB)."
    )


def file_record(path: Path, source_url: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if source_url is not None:
        record["source_url"] = source_url
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    configuration_path = repo_root / "task1/configs/task1.yaml"

    if not configuration_path.is_file():
        raise FileNotFoundError(configuration_path)

    # This occurs before any candidate image generation.
    record_source_in_configuration(configuration_path)

    external_repository = (
        artifacts_root / "external/pytorch-AdaIN-v0.0.0"
    )
    commit = prepare_external_repository(external_repository)
    models_directory = external_repository / "models"

    for filename, url in WEIGHT_URLS.items():
        download_file(url, models_directory / filename)

    source_record = {
        "method": "Adaptive Instance Normalization (AdaIN)",
        "method_paper": PAPER_URL,
        "original_authors_repository": {
            "url": ORIGINAL_REPOSITORY_URL,
            "framework": "Torch7",
            "role": "primary conceptual and original-code reference",
        },
        "executable_implementation": {
            "url": PYTORCH_REPOSITORY_URL,
            "status": "unofficial PyTorch reproduction",
            "release_url": RELEASE_URL,
            "release_tag": RELEASE_TAG,
            "commit": commit,
            "license": "MIT",
            "license_url": LICENSE_URL,
            "local_checkout": str(external_repository),
        },
        "approved_by_user_before_generation": True,
        "candidate_images_generated_at_record_time": 0,
        "configured_alpha": 0.8,
        "source_files": {
            "function.py": file_record(external_repository / "function.py"),
            "net.py": file_record(external_repository / "net.py"),
            "LICENSE": file_record(external_repository / "LICENSE"),
        },
        "pretrained_weights": {
            filename: file_record(models_directory / filename, url)
            for filename, url in WEIGHT_URLS.items()
        },
    }

    record_path = (
        repo_root / "task1/results/cue_conflict/adain_source.json"
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(source_record, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"PASS: wrote source and checksum record to {record_path}.")
    for filename, record in source_record["pretrained_weights"].items():
        print(
            f"{filename}: size={record['size_bytes']} bytes, "
            f"sha256={record['sha256']}"
        )
    print("PASS: AdaIN setup completed without generating any images.")


if __name__ == "__main__":
    main()
