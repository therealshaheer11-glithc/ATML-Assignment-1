from __future__ import annotations

import argparse

from torchvision import datasets

from task4.data.cifar10 import stratified_split_indices, write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    dataset = datasets.CIFAR10(
        root=args.data_root, train=True, transform=None, download=args.download
    )
    train_indices, validation_indices = stratified_split_indices(dataset.targets)
    write_split_manifest(args.output, dataset.targets, train_indices, validation_indices)
    print(f"Saved {len(train_indices)} training and {len(validation_indices)} validation indices")


if __name__ == "__main__":
    main()

