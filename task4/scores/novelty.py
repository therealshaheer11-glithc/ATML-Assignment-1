from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def msp_unknownness(logits: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.softmax(logits, dim=1).max(dim=1).values


def mls_unknownness(logits: torch.Tensor) -> torch.Tensor:
    return -logits.max(dim=1).values


def energy_unknownness(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


@dataclass(frozen=True)
class DiagonalMahalanobis:
    class_means: torch.Tensor
    shared_variance: torch.Tensor

    def unknownness(self, features: torch.Tensor) -> torch.Tensor:
        differences = features[:, None, :] - self.class_means[None, :, :]
        distances = (differences.square() / self.shared_variance[None, None, :]).sum(2)
        return distances.min(dim=1).values


def fit_diagonal_mahalanobis(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
    diagonal_epsilon: float = 1e-6,
) -> DiagonalMahalanobis:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Features and labels have incompatible shapes")
    means = []
    residuals = []
    for class_id in range(num_classes):
        class_features = features[labels == class_id]
        if class_features.numel() == 0:
            raise ValueError(f"Missing training features for class {class_id}")
        class_mean = class_features.mean(dim=0)
        means.append(class_mean)
        residuals.append(class_features - class_mean)
    shared_variance = torch.cat(residuals, dim=0).square().mean(dim=0)
    shared_variance = shared_variance + diagonal_epsilon
    return DiagonalMahalanobis(torch.stack(means), shared_variance)

