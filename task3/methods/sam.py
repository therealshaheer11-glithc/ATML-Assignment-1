"""Approved standard, non-adaptive SAM update with exact parameter restoration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SAMStepDiagnostics:
    first_loss: float
    perturbed_loss: float
    first_gradient_norm: float
    perturbation_norm: float
    second_gradient_norm: float
    second_gradient_norm_after_clipping: float
    second_gradient_was_clipped: bool

    def to_dict(self) -> dict:
        return asdict(self)


def total_gradient_norm(parameters: list[nn.Parameter]) -> float:
    squared_total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("A non-finite gradient was produced")
        squared_total += float(gradient.float().square().sum().item())
    norm = math.sqrt(squared_total)
    if not math.isfinite(norm):
        raise FloatingPointError("Total gradient norm is non-finite")
    return norm


def clip_gradients(
    parameters: list[nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    foreach: bool = False,
) -> tuple[float, float, bool]:
    if max_norm <= 0 or norm_type != 2.0 or foreach is not False:
        raise ValueError("Unexpected gradient-clipping configuration")

    pre_clip_norm = total_gradient_norm(parameters)
    torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=True,
        foreach=foreach,
    )
    post_clip_norm = total_gradient_norm(parameters)
    tolerance = max_norm * 1e-5
    if post_clip_norm > max_norm + tolerance:
        raise FloatingPointError(
            f"Gradient clipping failed: {post_clip_norm} > {max_norm}"
        )
    return pre_clip_norm, post_clip_norm, pre_clip_norm > max_norm


def sam_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_closure: Callable[[], Tensor],
    rho: float,
    max_gradient_norm: float,
) -> SAMStepDiagnostics:
    """Perform one standard SAM step on one already-created source batch.

    The first gradient is not clipped and is used only to construct the globally
    normalized perturbation. Parameters are restored from exact clones before the
    second-pass gradients are clipped and passed to AdamW.
    """
    if rho <= 0:
        raise ValueError("SAM rho must be positive")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("SAM received no trainable parameters")

    optimizer.zero_grad(set_to_none=True)
    first_loss = loss_closure()
    if first_loss.ndim != 0 or not torch.isfinite(first_loss):
        raise FloatingPointError("SAM first-pass loss is invalid")
    first_loss.backward()

    first_gradient_norm = total_gradient_norm(parameters)
    if first_gradient_norm <= 0.0:
        raise FloatingPointError("SAM first-pass gradient norm is zero")

    originals: list[tuple[nn.Parameter, Tensor]] = []
    perturbations: list[Tensor] = []
    scale = rho / first_gradient_norm

    # Construct and validate every perturbation before mutating any parameter. This
    # keeps a failed preflight calculation from leaving only part of the model at the
    # perturbed point.
    with torch.no_grad():
        for parameter in parameters:
            originals.append((parameter, parameter.detach().clone()))
            if parameter.grad is None:
                perturbation = torch.zeros_like(parameter)
            else:
                perturbation = parameter.grad.detach() * scale
            if not torch.isfinite(perturbation).all():
                raise FloatingPointError("SAM perturbation is non-finite")
            perturbations.append(perturbation)

        for (parameter, _original), perturbation in zip(
            originals,
            perturbations,
            strict=True,
        ):
            parameter.add_(perturbation)

    perturbation_norm = math.sqrt(
        sum(float(value.float().square().sum().item()) for value in perturbations)
    )
    if not math.isfinite(perturbation_norm):
        raise FloatingPointError("SAM perturbation norm is non-finite")
    perturbation_tolerance = max(1e-6, rho * 1e-4)
    if abs(perturbation_norm - rho) > perturbation_tolerance:
        raise FloatingPointError(
            "SAM perturbation does not have the approved global L2 radius: "
            f"{perturbation_norm} != {rho}"
        )

    optimizer.zero_grad(set_to_none=True)
    perturbed_loss: Tensor | None = None
    try:
        perturbed_loss = loss_closure()
        if perturbed_loss.ndim != 0 or not torch.isfinite(perturbed_loss):
            raise FloatingPointError("SAM perturbed loss is invalid")
        perturbed_loss.backward()
    finally:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original)

    if perturbed_loss is None:
        raise RuntimeError("SAM perturbed loss was not computed")

    (
        second_gradient_norm,
        second_gradient_norm_after_clipping,
        second_gradient_was_clipped,
    ) = clip_gradients(
        parameters,
        max_norm=max_gradient_norm,
        norm_type=2.0,
        foreach=False,
    )

    optimizer.step()

    return SAMStepDiagnostics(
        first_loss=float(first_loss.detach().item()),
        perturbed_loss=float(perturbed_loss.detach().item()),
        first_gradient_norm=first_gradient_norm,
        perturbation_norm=perturbation_norm,
        second_gradient_norm=second_gradient_norm,
        second_gradient_norm_after_clipping=second_gradient_norm_after_clipping,
        second_gradient_was_clipped=second_gradient_was_clipped,
    )
