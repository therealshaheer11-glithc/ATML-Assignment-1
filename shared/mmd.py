"""Three-kernel empirical MMD shared by Task 2 DAN and Task 3 DAN-DG."""

from __future__ import annotations

import torch
from torch import Tensor


def three_kernel_mmd(source: Tensor, target: Tensor) -> Tensor:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("MMD requires two feature matrices with the same width")
    both = torch.cat((source, target), dim=0).float()
    distances = torch.cdist(both, both).square()
    off_diagonal = distances[~torch.eye(len(both), dtype=torch.bool, device=both.device)]
    positive = off_diagonal[off_diagonal > 0]
    median = positive.detach().median() if positive.numel() else both.new_tensor(1.0)
    median = median.clamp_min(1e-12)
    kernel = sum(torch.exp(-distances / (factor * median)) for factor in (0.5, 1.0, 2.0))
    n = len(source)
    # Biased empirical squared MMD, including diagonal kernel terms.
    return kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()
