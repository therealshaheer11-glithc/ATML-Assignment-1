from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from task4.models import ProserResNet18


@dataclass(frozen=True)
class ProserLosses:
    total: torch.Tensor
    known_classification: torch.Tensor
    classifier_placeholder: torch.Tensor
    data_placeholder: torch.Tensor
    mix_lambda: float


def different_class_partners(labels: torch.Tensor) -> torch.Tensor:
    """Choose a random different-class partner for every item.

    Reuse is allowed when class counts prevent a different-class permutation. Every
    returned pair still satisfies the PA's different-class requirement.
    """
    partners = []
    for index, label in enumerate(labels):
        candidates = torch.nonzero(labels != label, as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("Manifold mixup requires at least two represented classes")
        choice = candidates[torch.randint(candidates.numel(), (), device=labels.device)]
        partners.append(choice)
    return torch.stack(partners)


def collapsed_dummy_logits(
    known_logits: torch.Tensor, dummy_logits: torch.Tensor
) -> torch.Tensor:
    max_dummy = dummy_logits.max(dim=1, keepdim=True).values
    return torch.cat([known_logits, max_dummy], dim=1)


def classifier_placeholder_losses(
    known_logits: torch.Tensor,
    dummy_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_logits = torch.cat([known_logits, dummy_logits], dim=1)
    known_classification = F.cross_entropy(all_logits, labels)

    masked_known = known_logits.clone()
    masked_known.scatter_(1, labels[:, None], torch.finfo(masked_known.dtype).min)
    collapsed = collapsed_dummy_logits(masked_known, dummy_logits)
    dummy_target = torch.full_like(labels, known_logits.shape[1])
    classifier_placeholder = F.cross_entropy(collapsed, dummy_target)
    combined = known_classification + beta * classifier_placeholder
    return combined, known_classification, classifier_placeholder


def proser_training_loss(
    model: ProserResNet18,
    images: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 0.1,
) -> ProserLosses:
    if images.shape[0] < 4 or images.shape[0] % 2:
        raise ValueError("PROSER requires an even mini-batch with at least four examples")
    half = images.shape[0] // 2

    classifier_images = images[:half]
    classifier_labels = labels[:half]
    known_logits, dummy_logits = model(classifier_images)
    classifier_total, known_ce, classifier_placeholder = classifier_placeholder_losses(
        known_logits, dummy_logits, classifier_labels, beta=beta
    )

    mix_images = images[half:]
    mix_labels = labels[half:]
    layer2 = model.forward_to_layer2(mix_images)
    partners = different_class_partners(mix_labels)
    distribution = torch.distributions.Beta(
        torch.tensor(2.0, device=images.device),
        torch.tensor(2.0, device=images.device),
    )
    mix_lambda_tensor = distribution.sample()
    mixed = mix_lambda_tensor * layer2 + (1.0 - mix_lambda_tensor) * layer2[partners]
    mixed_features = model.forward_from_layer2(mixed)
    mixed_known, mixed_dummy = model.classify_features(mixed_features)
    mixed_collapsed = collapsed_dummy_logits(mixed_known, mixed_dummy)
    dummy_target = torch.full_like(mix_labels, mixed_known.shape[1])
    data_placeholder = F.cross_entropy(mixed_collapsed, dummy_target)

    total = classifier_total + gamma * data_placeholder
    return ProserLosses(
        total=total,
        known_classification=known_ce,
        classifier_placeholder=classifier_placeholder,
        data_placeholder=data_placeholder,
        mix_lambda=float(mix_lambda_tensor.detach().cpu()),
    )


def calibrate_dummy_bias(
    known_logits: torch.Tensor, dummy_logits: torch.Tensor, quantile: float = 0.95
) -> float:
    """Choose a bias so 95% of known validation items beat the strongest dummy."""
    margin = dummy_logits.max(dim=1).values - known_logits.max(dim=1).values
    return -float(torch.quantile(margin, quantile).item())


def placeholder_unknownness(
    known_logits: torch.Tensor,
    dummy_logits: torch.Tensor,
    bias: float,
    temperature: float = 1024.0,
) -> torch.Tensor:
    """Reference PROSER delta-probability score; larger means more unknown."""
    max_dummy = dummy_logits.max(dim=1, keepdim=True).values + bias
    augmented = torch.cat([known_logits, max_dummy], dim=1) / temperature
    probabilities = F.softmax(augmented, dim=1)
    return probabilities[:, -1] - probabilities[:, :-1].max(dim=1).values

