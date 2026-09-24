"""Deterministic common local-sharpness proxy for Task 3."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from task3.data import SEED, SOURCES


SHARPNESS_PER_DOMAIN = 32
SHARPNESS_RADIUS = 0.05


def _seeded_subset(
    records: Sequence[dict],
    count: int,
    generator: np.random.Generator,
) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Cannot select {count} records from only {len(records)}")
    indices = np.sort(generator.choice(len(records), size=count, replace=False))
    return [dict(records[int(index)]) for index in indices]


def select_sharpness_records(
    protocol: dict,
    *,
    seed: int = SEED,
    per_domain: int = SHARPNESS_PER_DOMAIN,
) -> dict[str, list[dict]]:
    """Select 32 validation examples per domain with one independent seeded stream."""
    generator = np.random.default_rng(seed)
    return {
        domain: _seeded_subset(
            protocol["source_splits"][domain]["validation"],
            per_domain,
            generator,
        )
        for domain in SOURCES
    }


def global_gradient_norm(parameters: Sequence[nn.Parameter]) -> Tensor:
    squared = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().double().square().sum()
    return squared.sqrt()


def common_sharpness_proxy(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    *,
    radius: float = SHARPNESS_RADIUS,
) -> dict[str, float]:
    """Measure the prescribed one-step CE increase and restore parameters exactly."""
    if radius <= 0:
        raise ValueError("Sharpness radius must be positive")
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Sharpness model has no trainable parameters")

    originals = [parameter.detach().clone() for parameter in parameters]
    model.zero_grad(set_to_none=True)
    logits, _features = model(images)
    base_loss = nn.functional.cross_entropy(logits, labels, reduction="mean")
    if not torch.isfinite(base_loss):
        raise FloatingPointError("Base sharpness loss is non-finite")
    base_loss.backward()
    gradient_norm = global_gradient_norm(parameters)
    if not torch.isfinite(gradient_norm) or gradient_norm.item() <= 0:
        raise FloatingPointError("Sharpness gradient norm is invalid")

    scale = radius / gradient_norm
    perturbations: list[Tensor] = []
    try:
        with torch.no_grad():
            for parameter in parameters:
                perturbation = (
                    torch.zeros_like(parameter)
                    if parameter.grad is None
                    else parameter.grad.detach() * scale.to(parameter.dtype)
                )
                parameter.add_(perturbation)
                perturbations.append(perturbation)

        with torch.no_grad():
            perturbed_logits, _features = model(images)
            perturbed_loss = nn.functional.cross_entropy(
                perturbed_logits, labels, reduction="mean"
            )
        if not torch.isfinite(perturbed_loss):
            raise FloatingPointError("Perturbed sharpness loss is non-finite")
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
        model.zero_grad(set_to_none=True)

    if any(
        not torch.equal(parameter.detach(), original)
        for parameter, original in zip(parameters, originals, strict=True)
    ):
        raise RuntimeError("Sharpness diagnostic failed to restore parameters exactly")

    perturbation_norm = torch.sqrt(
        sum(perturbation.detach().double().square().sum() for perturbation in perturbations)
    )
    delta = perturbed_loss - base_loss.detach()
    return {
        "base_cross_entropy": float(base_loss.detach().item()),
        "perturbed_cross_entropy": float(perturbed_loss.item()),
        "delta_sharp": float(delta.item()),
        "gradient_norm": float(gradient_norm.item()),
        "radius": float(radius),
        "actual_perturbation_norm": float(perturbation_norm.item()),
    }
