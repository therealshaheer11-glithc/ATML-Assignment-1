"""Pairwise source-domain MMD for the approved DAN-DG objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from shared.mmd import l2_normalize_alignment_features, three_kernel_mmd


SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class PairwiseMMDDiagnostics:
    pair: str
    mmd_squared: float
    median_squared_distance: float
    bandwidths: tuple[float, float, float]
    distinct_pair_count: int
    off_diagonal_zero_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def dan_dg_pairwise_mmd(
    features: Tensor,
    batch_per_domain: int = 8,
) -> tuple[Tensor, tuple[PairwiseMMDDiagnostics, ...]]:
    """Average the locked MMD over all three unordered source-domain pairs.

    `features` must be ordered Photo, Art Painting, Cartoon with the same number of
    examples per domain. The classifier must receive the original features; only the
    tensors passed into this function are L2-normalized for MMD.
    """
    expected_rows = len(SOURCE_DOMAINS) * batch_per_domain
    if features.ndim != 2:
        raise ValueError("DAN-DG features must be a rank-2 matrix")
    if features.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} ordered source features, got {features.shape[0]}"
        )
    if not torch.isfinite(features).all():
        raise FloatingPointError("DAN-DG received non-finite features")

    raw_domains = features.split(batch_per_domain, dim=0)
    if len(raw_domains) != len(SOURCE_DOMAINS):
        raise RuntimeError("Could not split DAN-DG features into three source domains")

    normalized_domains = tuple(
        l2_normalize_alignment_features(domain_features)
        for domain_features in raw_domains
    )

    values: list[Tensor] = []
    diagnostics: list[PairwiseMMDDiagnostics] = []

    for left_index, right_index in PAIR_INDICES:
        value, details = three_kernel_mmd(
            normalized_domains[left_index],
            normalized_domains[right_index],
        )
        values.append(value)
        diagnostics.append(
            PairwiseMMDDiagnostics(
                pair=(
                    f"{SOURCE_DOMAINS[left_index]}__{SOURCE_DOMAINS[right_index]}"
                ),
                mmd_squared=float(value.detach().item()),
                median_squared_distance=details.median_squared_distance,
                bandwidths=details.bandwidths,
                distinct_pair_count=details.distinct_pair_count,
                off_diagonal_zero_count=details.off_diagonal_zero_count,
            )
        )

    mean_mmd = torch.stack(values).mean()
    if not torch.isfinite(mean_mmd):
        raise FloatingPointError("Mean DAN-DG MMD is non-finite")

    return mean_mmd, tuple(diagnostics)

