"""DANN/CDAN discriminators and gradient reversal."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.autograd import Function


class _ReverseGradient(Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return -ctx.strength * gradient, None


def reverse_gradient(inputs: Tensor, strength: float) -> Tensor:
    return _ReverseGradient.apply(inputs, strength)


def reversal_strength(progress: float, maximum: float = 1.0) -> float:
    if not 0 <= progress <= 1:
        raise ValueError("Training progress must lie in [0,1]")
    return maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def conditional_features(features: Tensor, logits: Tensor) -> Tensor:
    """Flatten f outer-product softmax(C(f)) without detaching either input."""
    probabilities = logits.softmax(dim=1)
    return torch.bmm(features.unsqueeze(2), probabilities.unsqueeze(1)).flatten(1)


class DomainDiscriminator(nn.Sequential):
    def __init__(self, conditional: bool) -> None:
        super().__init__(
            nn.Linear(512 * (7 if conditional else 1), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
