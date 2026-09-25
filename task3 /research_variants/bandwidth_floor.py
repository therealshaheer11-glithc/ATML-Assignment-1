"""DAN-DG with the approved initialization-anchored bandwidth floor.

This module is supplementary. It intentionally does not modify shared/mmd.py or the
prescribed DAN-DG implementation. The sole mathematical change is
`effective_median = max(current_batch_median, frozen_pair_floor)`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median

import torch
from torch import Tensor

from shared.mmd import KERNEL_FACTORS, l2_normalize_alignment_features
from task3.methods.dan_dg import PAIR_INDICES, SOURCE_DOMAINS


PAIR_NAMES = tuple(
    f"{SOURCE_DOMAINS[left]}__{SOURCE_DOMAINS[right]}"
    for left, right in PAIR_INDICES
)


@dataclass(frozen=True)
class FloorMMDDiagnostics:
    pair: str
    mmd_squared: float
    current_median_squared_distance: float
    effective_median_squared_distance: float
    frozen_floor_squared_distance: float
    floor_activated: bool
    bandwidths: tuple[float, float, float]
    distinct_pair_count: int
    off_diagonal_zero_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _distance_matrix(features: Tensor) -> Tensor:
    values = features.float()
    differences = values[:, None, :] - values[None, :, :]
    squared = differences.square().sum(dim=-1)
    if not torch.isfinite(squared).all():
        raise FloatingPointError("Pairwise squared distances are non-finite")
    return squared


def _validate_pair(source: Tensor, target: Tensor) -> None:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("MMD inputs must be rank-2 feature matrices")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source feature widths differ")
    if source.shape[0] < 2 or target.shape[0] < 2:
        raise ValueError("MMD requires at least two examples per side")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise FloatingPointError("MMD received non-finite features")


def current_pair_median(source: Tensor, target: Tensor) -> tuple[Tensor, int, int]:
    """Return the exact locked current-batch bandwidth statistic."""
    _validate_pair(source, target)
    combined = torch.cat((source, target), dim=0)
    distances = _distance_matrix(combined)
    upper_mask = torch.triu(
        torch.ones_like(distances, dtype=torch.bool), diagonal=1
    )
    distinct = distances[upper_mask]
    if distinct.numel() == 0:
        raise RuntimeError("No distinct pairs are available for bandwidth calibration")
    value = distinct.detach().median()
    zero_count = int((distinct == 0).sum().item())
    if not torch.isfinite(value) or value.item() <= 0.0:
        raise FloatingPointError(
            "Current MMD median is non-positive or non-finite; "
            f"distinct_pairs={distinct.numel()}, off_diagonal_zeros={zero_count}"
        )
    return value, int(distinct.numel()), zero_count


def three_kernel_mmd_with_floor(
    source: Tensor,
    target: Tensor,
    *,
    bandwidth_floor: float,
) -> tuple[Tensor, FloorMMDDiagnostics]:
    """Apply the locked kernel and estimator with one declared median floor."""
    if not torch.isfinite(torch.tensor(bandwidth_floor)) or bandwidth_floor <= 0.0:
        raise ValueError("The frozen bandwidth floor must be positive and finite")

    _validate_pair(source, target)
    combined = torch.cat((source, target), dim=0)
    distances = _distance_matrix(combined)
    upper_mask = torch.triu(
        torch.ones_like(distances, dtype=torch.bool), diagonal=1
    )
    distinct = distances[upper_mask]
    current = distinct.detach().median()
    zeros = int((distinct == 0).sum().item())
    if not torch.isfinite(current) or current.item() <= 0.0:
        raise FloatingPointError(
            "Current MMD median is non-positive or non-finite; "
            f"distinct_pairs={distinct.numel()}, off_diagonal_zeros={zeros}"
        )

    floor_tensor = current.new_tensor(float(bandwidth_floor))
    effective = torch.maximum(current, floor_tensor).detach()
    activated = bool((current < floor_tensor).item())

    kernels: list[Tensor] = []
    bandwidths: list[float] = []
    for factor in KERNEL_FACTORS:
        bandwidth = factor * effective
        kernels.append(torch.exp(-distances / (2.0 * bandwidth)))
        bandwidths.append(float(bandwidth.item()))
    kernel = torch.stack(kernels, dim=0).sum(dim=0)

    ns = source.shape[0]
    value = (
        kernel[:ns, :ns].mean()
        + kernel[ns:, ns:].mean()
        - 2.0 * kernel[:ns, ns:].mean()
    )
    if not torch.isfinite(value):
        raise FloatingPointError("Bandwidth-floor MMD is non-finite")

    diagnostics = FloorMMDDiagnostics(
        pair="",
        mmd_squared=float(value.detach().item()),
        current_median_squared_distance=float(current.item()),
        effective_median_squared_distance=float(effective.item()),
        frozen_floor_squared_distance=float(floor_tensor.item()),
        floor_activated=activated,
        bandwidths=tuple(bandwidths),  # type: ignore[arg-type]
        distinct_pair_count=int(distinct.numel()),
        off_diagonal_zero_count=zeros,
    )
    return value, diagnostics


def dan_dg_pairwise_mmd_with_floor(
    features: Tensor,
    floors: dict[str, float],
    *,
    batch_per_domain: int = 8,
) -> tuple[Tensor, tuple[FloorMMDDiagnostics, ...]]:
    expected_rows = len(SOURCE_DOMAINS) * batch_per_domain
    if features.ndim != 2 or features.shape[0] != expected_rows:
        raise ValueError(
            f"Expected a {expected_rows}-row feature matrix, got {tuple(features.shape)}"
        )
    if set(floors) != set(PAIR_NAMES):
        raise ValueError("Frozen floor mapping does not contain the three source pairs")
    if not torch.isfinite(features).all():
        raise FloatingPointError("DAN-DG received non-finite features")

    domains = tuple(
        l2_normalize_alignment_features(block)
        for block in features.split(batch_per_domain, dim=0)
    )
    values: list[Tensor] = []
    diagnostics: list[FloorMMDDiagnostics] = []
    for left, right in PAIR_INDICES:
        pair = f"{SOURCE_DOMAINS[left]}__{SOURCE_DOMAINS[right]}"
        value, details = three_kernel_mmd_with_floor(
            domains[left], domains[right], bandwidth_floor=floors[pair]
        )
        values.append(value)
        diagnostics.append(
            FloorMMDDiagnostics(**{**details.to_dict(), "pair": pair})
        )

    mean_value = torch.stack(values).mean()
    if not torch.isfinite(mean_value):
        raise FloatingPointError("Mean bandwidth-floor MMD is non-finite")
    return mean_value, tuple(diagnostics)


def reduce_calibration_medians(values: dict[str, list[float]]) -> dict[str, dict]:
    """Reduce all 235 initialization medians with no tunable multiplier."""
    if set(values) != set(PAIR_NAMES):
        raise ValueError("Calibration values do not contain all three source pairs")
    reduced: dict[str, dict] = {}
    for pair in PAIR_NAMES:
        samples = [float(value) for value in values[pair]]
        if len(samples) != 235:
            raise ValueError(f"Expected 235 calibration medians for {pair}")
        if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in samples):
            raise FloatingPointError(f"Invalid calibration median for {pair}")
        ordered = sorted(samples)
        floor = float(median(ordered))
        reduced[pair] = {
            "floor_squared_distance": floor,
            "sample_count": len(samples),
            "minimum": ordered[0],
            "maximum": ordered[-1],
            "mean": float(sum(ordered) / len(ordered)),
            "median": floor,
            "floor_multiplier": 1.0,
        }
    return reduced

