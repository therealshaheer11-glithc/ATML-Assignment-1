"""Locked three-kernel MMD for the corrected Task 2/3 protocol."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


KERNEL_FACTORS = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class MMDDiagnostics:
    median_squared_distance: float
    bandwidths: tuple[float, float, float]
    distinct_pair_count: int
    off_diagonal_zero_count: int


def _validate_features(source: Tensor, target: Tensor) -> None:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("MMD inputs must be rank-2 feature matrices")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target feature widths differ")
    if source.shape[0] < 2 or target.shape[0] < 2:
        raise ValueError("MMD requires at least two source and two target features")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise FloatingPointError("MMD received a non-finite feature")


def _squared_distances(features: Tensor) -> Tensor:
    # Compute squared Euclidean distance directly. This avoids taking a square root
    # only to square it again and makes every diagonal self-distance exactly zero.
    values = features.float()
    differences = values[:, None, :] - values[None, :, :]
    squared = differences.square().sum(dim=-1)
    if not torch.isfinite(squared).all():
        raise FloatingPointError("Pairwise squared distances are non-finite")
    return squared


def three_kernel_mmd(source: Tensor, target: Tensor) -> tuple[Tensor, MMDDiagnostics]:
    """Return the locked empirical mean-embedding MMD and batch diagnostics.

    Bandwidth median:
      * combined source/target batch;
      * each distinct pair once (strict upper triangle);
      * diagonal self-distances excluded;
      * off-diagonal zeros retained;
      * median detached from autograd.

    Estimator:
      mean(K_ss) + mean(K_tt) - 2*mean(K_st), including K_ss/K_tt diagonals.

    Kernel:
      sum_b exp(-d^2/(2b)), b in {0.5m, m, 2m}.
    """

    _validate_features(source, target)
    combined = torch.cat((source, target), dim=0)
    distances = _squared_distances(combined)

    upper_mask = torch.triu(
        torch.ones_like(distances, dtype=torch.bool), diagonal=1
    )
    distinct = distances[upper_mask]
    if distinct.numel() == 0:
        raise RuntimeError("No distinct feature pairs are available for the bandwidth")

    median = distinct.detach().median()
    if not torch.isfinite(median) or median.item() <= 0.0:
        zeros = int((distinct == 0).sum().item())
        raise FloatingPointError(
            "MMD bandwidth median is non-positive or non-finite; "
            f"distinct_pairs={distinct.numel()}, off_diagonal_zeros={zeros}"
        )

    kernels = []
    bandwidth_values: list[float] = []
    for factor in KERNEL_FACTORS:
        bandwidth = factor * median
        kernels.append(torch.exp(-distances / (2.0 * bandwidth)))
        bandwidth_values.append(float(bandwidth.item()))
    kernel = torch.stack(kernels, dim=0).sum(dim=0)

    ns = source.shape[0]
    mmd_squared = (
        kernel[:ns, :ns].mean()
        + kernel[ns:, ns:].mean()
        - 2.0 * kernel[:ns, ns:].mean()
    )
    if not torch.isfinite(mmd_squared):
        raise FloatingPointError("MMD result is non-finite")

    diagnostics = MMDDiagnostics(
        median_squared_distance=float(median.item()),
        bandwidths=tuple(bandwidth_values),  # type: ignore[arg-type]
        distinct_pair_count=int(distinct.numel()),
        off_diagonal_zero_count=int((distinct == 0).sum().item()),
    )
    return mmd_squared, diagnostics
