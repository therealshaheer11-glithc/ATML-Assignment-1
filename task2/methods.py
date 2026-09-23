"""Required DANN/CDAN components."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.autograd import Function

from task2.model import FEATURE_WIDTH, NUM_CLASSES


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, values: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return -ctx.strength * gradient, None


def reverse_gradient(values: Tensor, strength: float) -> Tensor:
    return _GradientReversal.apply(values, strength)


def grl_strength(progress: float) -> float:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("GRL progress must be in [0,1]")
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def conditional_features(features: Tensor, logits: Tensor) -> Tensor:
    probabilities = logits.softmax(dim=1)
    # No detach: gradients must pass through both features and probabilities.
    return torch.bmm(features.unsqueeze(2), probabilities.unsqueeze(1)).flatten(1)


class DomainDiscriminator(nn.Sequential):
    def __init__(self, conditional: bool) -> None:
        input_width = FEATURE_WIDTH * (NUM_CLASSES if conditional else 1)
        super().__init__(
            nn.Linear(input_width, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

