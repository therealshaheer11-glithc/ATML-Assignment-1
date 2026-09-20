
"""Deterministic common-image construction and Task 1 interventions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as torch_functional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional


IMAGE_SIZE = 224
PATCH_GRID_SIZE = 4
PATCH_SIZE = IMAGE_SIZE // PATCH_GRID_SIZE
ASSIGNMENT_SEED = 6304
TRANSLATION_DIRECTIONS = ("up", "down", "left", "right")


def common_rgb_tensor(image: Image.Image) -> Tensor:
    """Convert an image into the common 224x224 RGB tensor in [0, 1]."""

    rgb_image = image.convert("RGB")
    resized = vision_functional.resize(
        rgb_image,
        [IMAGE_SIZE, IMAGE_SIZE],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    tensor = vision_functional.pil_to_tensor(resized).to(torch.float32)
    return (tensor / 255.0).contiguous()


def load_common_rgb_tensor(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        return common_rgb_tensor(image)


def common_tensor_to_pil(image: Tensor) -> Image.Image:
    _validate_common_image(image)
    return vision_functional.to_pil_image(image.cpu())


def _validate_common_image(image: Tensor) -> None:
    if tuple(image.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            "Expected a [3, 224, 224] common RGB tensor; "
            f"received {tuple(image.shape)}."
        )

    if not image.is_floating_point():
        raise TypeError("Common images must be floating-point tensors.")

    minimum = float(image.min())
    maximum = float(image.max())

    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            f"Expected image values in [0, 1]; observed [{minimum}, {maximum}]."
        )


def grayscale(image: Tensor) -> Tensor:
    """Remove chromatic information while preserving three RGB channels."""

    _validate_common_image(image)
    transformed = vision_functional.rgb_to_grayscale(
        image,
        num_output_channels=3,
    )
    return transformed.contiguous()


def hue_rotate(
    image: Tensor,
    angle_degrees: float = 90.0,
) -> Tensor:
    """Apply a fixed HSV hue rotation while retaining geometry."""

    _validate_common_image(image)

    if not -180.0 <= angle_degrees <= 180.0:
        raise ValueError("Hue rotation must lie between -180 and 180 degrees.")

    hue_factor = angle_degrees / 360.0
    transformed = vision_functional.adjust_hue(image, hue_factor)
    return transformed.clamp(0.0, 1.0).contiguous()


def reflection_translate(
    image: Tensor,
    displacement: int,
    direction: str,
) -> Tensor:
    """Translate using reflection padding followed by a shifted crop."""

    _validate_common_image(image)

    if direction not in TRANSLATION_DIRECTIONS:
        raise ValueError(
            f"Unknown direction {direction!r}; "
            f"choose from {TRANSLATION_DIRECTIONS}."
        )

    if displacement < 0:
        raise ValueError("Displacement must be non-negative.")

    if displacement >= IMAGE_SIZE:
        raise ValueError("Displacement must be smaller than the image size.")

    if displacement == 0:
        return image.clone()

    padded = torch_functional.pad(
        image,
        (
            displacement,
            displacement,
            displacement,
            displacement,
        ),
        mode="reflect",
    )

    crop_top = displacement
    crop_left = displacement

    if direction == "up":
        crop_top = 2 * displacement
    elif direction == "down":
        crop_top = 0
    elif direction == "left":
        crop_left = 2 * displacement
    elif direction == "right":
        crop_left = 0

    translated = padded[
        :,
        crop_top : crop_top + IMAGE_SIZE,
        crop_left : crop_left + IMAGE_SIZE,
    ]

    return translated.contiguous()


def make_patch_permutation(
    image_identifier: int,
    seed: int = ASSIGNMENT_SEED,
    grid_size: int = PATCH_GRID_SIZE,
) -> tuple[int, ...]:
    """Generate one reproducible non-identity permutation for an image."""

    if image_identifier < 0:
        raise ValueError("Image identifier must be non-negative.")

    if grid_size <= 1:
        raise ValueError("Grid size must be greater than one.")

    number_of_patches = grid_size * grid_size
    identity = np.arange(number_of_patches)

    # Combining the assignment seed with the official image identifier makes
    # the permutation independent of DataLoader ordering or model choice.
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(image_identifier)]
    )
    generator = np.random.default_rng(seed_sequence)

    for _ in range(100):
        permutation = generator.permutation(number_of_patches)

        if not np.array_equal(permutation, identity):
            return tuple(map(int, permutation))

    raise RuntimeError("Failed to generate a non-identity permutation.")


def patch_shuffle(
    image: Tensor,
    permutation: Sequence[int],
    grid_size: int = PATCH_GRID_SIZE,
) -> Tensor:
    """Rearrange a pixel-space grid without changing patch contents."""

    _validate_common_image(image)

    if IMAGE_SIZE % grid_size != 0:
        raise ValueError("Grid size must divide the 224-pixel image size.")

    number_of_patches = grid_size * grid_size
    permutation_tuple = tuple(map(int, permutation))

    if sorted(permutation_tuple) != list(range(number_of_patches)):
        raise ValueError("Permutation must contain each patch index once.")

    if permutation_tuple == tuple(range(number_of_patches)):
        raise ValueError("Patch permutation must be non-identity.")

    channels, height, width = image.shape
    patch_height = height // grid_size
    patch_width = width // grid_size

    patches = (
        image.reshape(
            channels,
            grid_size,
            patch_height,
            grid_size,
            patch_width,
        )
        .permute(1, 3, 0, 2, 4)
        .contiguous()
        .reshape(
            number_of_patches,
            channels,
            patch_height,
            patch_width,
        )
    )

    order = torch.tensor(
        permutation_tuple,
        dtype=torch.long,
        device=image.device,
    )
    shuffled_patches = patches.index_select(0, order)

    shuffled_image = (
        shuffled_patches.reshape(
            grid_size,
            grid_size,
            channels,
            patch_height,
            patch_width,
        )
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .reshape(channels, height, width)
    )

    return shuffled_image
