"""High-value checks for the corrected protocol's failure-prone conventions."""

from __future__ import annotations

import math

import torch
from torch import nn

from shared.mmd import l2_normalize_alignment_features, three_kernel_mmd
from task2.methods import conditional_features, grl_strength
from task2.model import assert_batchnorm_unchanged, batchnorm_buffers, freeze_batchnorm_statistics
from task2.train import clip_gradients


def test_mmd_keeps_off_diagonal_zero_for_bandwidth() -> None:
    # Rows 0 and 1 are different examples with identical features. Their zero distance
    # must remain in the six distinct-pair median candidates.
    source = torch.tensor([[0.0], [0.0]])
    target = torch.tensor([[2.0], [4.0]])
    value, diagnostics = three_kernel_mmd(source, target)
    assert torch.isfinite(value)
    assert diagnostics.distinct_pair_count == 6
    assert diagnostics.off_diagonal_zero_count == 1
    # Unique squared distances: [0,4,16,4,16,4], lower median = 4.
    assert diagnostics.median_squared_distance == 4.0
    assert diagnostics.bandwidths == (2.0, 4.0, 8.0)


def test_mmd_is_literal_empirical_mean_embedding_expression() -> None:
    source = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[2.0], [3.0]])
    actual, diagnostics = three_kernel_mmd(source, target)
    all_features = torch.cat((source, target))
    d2 = torch.cdist(all_features, all_features).square()
    expected_kernel = sum(
        torch.exp(-d2 / (2.0 * factor * diagnostics.median_squared_distance))
        for factor in (0.5, 1.0, 2.0)
    )
    expected = (
        expected_kernel[:2, :2].mean()
        + expected_kernel[2:, 2:].mean()
        - 2.0 * expected_kernel[:2, 2:].mean()
    )
    assert torch.allclose(actual, expected)


def test_zero_median_stops_instead_of_filtering_zeros() -> None:
    source = torch.zeros(2, 3)
    target = torch.zeros(2, 3)
    try:
        three_kernel_mmd(source, target)
    except FloatingPointError as error:
        assert "median" in str(error)
    else:
        raise AssertionError("A zero bandwidth median must stop the run")


def test_dan_mmd_features_are_l2_normalized_without_an_epsilon() -> None:
    features = torch.tensor([[3.0, 4.0], [0.0, 2.0]], requires_grad=True)
    normalized = l2_normalize_alignment_features(features)
    assert torch.allclose(torch.linalg.vector_norm(normalized, dim=1), torch.ones(2))
    assert torch.allclose(normalized[0], torch.tensor([0.6, 0.8]))
    normalized.sum().backward()
    assert features.grad is not None

    try:
        l2_normalize_alignment_features(torch.zeros(2, 3))
    except FloatingPointError as error:
        assert "non-positive norm" in str(error)
    else:
        raise AssertionError("A zero feature norm must stop the run")


def test_batchnorm_running_buffers_are_frozen_but_affine_is_trainable() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU())
    model.train()
    before = batchnorm_buffers(model)
    freeze_batchnorm_statistics(model)
    bn = model[1]
    assert bn.training is False
    assert bn.weight.requires_grad and bn.bias.requires_grad
    model(torch.randn(8, 4)).sum().backward()
    assert_batchnorm_unchanged(model, before)


def test_cdan_width_and_no_detachment() -> None:
    features = torch.randn(5, 512, requires_grad=True)
    logits = torch.randn(5, 7, requires_grad=True)
    normalized = l2_normalize_alignment_features(features)
    conditioned = conditional_features(normalized, logits)
    assert conditioned.shape == (5, 3584)
    conditioned.sum().backward()
    assert features.grad is not None
    assert logits.grad is not None


def test_adversarial_features_are_l2_normalized_without_detachment() -> None:
    features = torch.randn(5, 512, requires_grad=True)
    normalized = l2_normalize_alignment_features(features)
    assert torch.allclose(
        torch.linalg.vector_norm(normalized, ord=2, dim=1),
        torch.ones(5),
        atol=1e-6,
    )
    normalized.square().sum().backward()
    assert features.grad is not None


def test_required_grl_schedule() -> None:
    assert grl_strength(0.0) == 0.0
    assert math.isclose(grl_strength(1.0), 2.0 / (1.0 + math.exp(-10.0)) - 1.0)


def test_global_gradient_clipping_at_twenty() -> None:
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([30.0, 40.0])
    before, after, clipped = clip_gradients(
        [parameter], max_norm=20.0, norm_type=2.0, foreach=False
    )
    assert math.isclose(before, 50.0)
    assert math.isclose(after, 20.0, rel_tol=1e-6)
    assert clipped is True
