from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from task4.config import load_config
from task4.data.cifar10 import stratified_split_indices, training_transform
from task4.data.cifar100_unknowns import FAR_UNKNOWN_CLASSES, NEAR_UNKNOWN_CLASSES
from task4.evaluation.metrics import known_unknown_metrics, validation_threshold
from task4.methods.proser import (
    calibrate_dummy_bias,
    classifier_placeholder_losses,
    different_class_partners,
    placeholder_unknownness,
    proser_training_loss,
)
from task4.models import CifarResNet18, ProserResNet18
from task4.scores import (
    energy_unknownness,
    fit_diagonal_mahalanobis,
    mls_unknownness,
    msp_unknownness,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_locked_configs_validate() -> None:
    for method in ("vanilla", "gcsc", "proser"):
        config = load_config(ROOT / "configs" / f"{method}.yaml")
        assert config["method"] == method


def test_stratified_split_is_balanced_disjoint_and_deterministic() -> None:
    labels = np.repeat(np.arange(10), 100)
    train_a, validation_a = stratified_split_indices(labels)
    train_b, validation_b = stratified_split_indices(labels)
    assert (train_a, validation_a) == (train_b, validation_b)
    assert len(train_a) == 900 and len(validation_a) == 100
    assert set(train_a).isdisjoint(validation_a)
    assert np.bincount(labels[validation_a], minlength=10).tolist() == [10] * 10


def test_gcsc_randaugment_order_matches_pa() -> None:
    names = [type(item).__name__ for item in training_transform("gcsc").transforms]
    assert names == [
        "RandomCrop",
        "RandomHorizontalFlip",
        "RandAugment",
        "ToTensor",
        "Normalize",
    ]


def test_unknown_groups_are_exact_and_disjoint() -> None:
    assert NEAR_UNKNOWN_CLASSES == (
        "bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"
    )
    assert FAR_UNKNOWN_CLASSES == (
        "bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"
    )
    assert set(NEAR_UNKNOWN_CLASSES).isdisjoint(FAR_UNKNOWN_CLASSES)


def test_cifar_resnet_stem_and_shapes() -> None:
    model = CifarResNet18()
    assert model.backbone.conv1.kernel_size == (3, 3)
    assert model.backbone.conv1.stride == (1, 1)
    assert isinstance(model.backbone.maxpool, nn.Identity)
    logits, features = model(torch.randn(2, 3, 32, 32), return_features=True)
    assert logits.shape == (2, 10)
    assert features.shape == (2, 512)


def test_proser_has_five_dummies_and_known_head_is_reused() -> None:
    known = CifarResNet18()
    original_weight = known.backbone.fc.weight.detach().clone()
    model = ProserResNet18(known)
    known_logits, dummy_logits = model(torch.randn(2, 3, 32, 32))
    assert known_logits.shape == (2, 10)
    assert dummy_logits.shape == (2, 5)
    assert torch.equal(model.known_model.backbone.fc.weight, original_weight)


def test_different_class_partners_never_match_labels() -> None:
    torch.manual_seed(6304)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    partners = different_class_partners(labels)
    assert torch.all(labels != labels[partners])


def test_classifier_placeholder_masks_true_class_and_targets_dummy() -> None:
    known = torch.tensor([[10.0, 1.0, 0.0], [1.0, 9.0, 0.0]])
    dummy = torch.tensor([[2.0, 3.0], [3.0, 2.0]])
    labels = torch.tensor([0, 1])
    total, known_ce, placeholder = classifier_placeholder_losses(known, dummy, labels)
    assert torch.isfinite(total)
    assert torch.allclose(total, known_ce + placeholder)
    assert placeholder < 0.5


def test_complete_proser_loss_backpropagates_through_both_placeholders() -> None:
    torch.manual_seed(6304)
    model = ProserResNet18(CifarResNet18())
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    losses = proser_training_loss(model, images, labels)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert model.dummy_classifier.weight.grad is not None
    assert torch.isfinite(model.dummy_classifier.weight.grad).all()


def test_required_posthoc_score_formulas() -> None:
    logits = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
    assert torch.allclose(mls_unknownness(logits), torch.tensor([-2.0, 1.0]))
    assert torch.allclose(energy_unknownness(logits), -torch.logsumexp(logits, dim=1))
    assert torch.allclose(msp_unknownness(logits), 1 - torch.softmax(logits, dim=1).max(1).values)


def test_diagonal_mahalanobis_uses_shared_within_class_variance() -> None:
    features = torch.tensor([[0.0, 0.0], [2.0, 2.0], [10.0, 0.0], [12.0, 2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    model = fit_diagonal_mahalanobis(features, labels, num_classes=2)
    assert torch.allclose(model.class_means, torch.tensor([[1.0, 1.0], [11.0, 1.0]]))
    assert torch.allclose(model.shared_variance, torch.tensor([1.000001, 1.000001]))
    assert torch.allclose(model.unknownness(torch.tensor([[1.0, 1.0]])), torch.tensor([0.0]))


def test_validation_threshold_and_metrics_use_larger_as_unknown() -> None:
    known = np.arange(100, dtype=float)
    threshold = validation_threshold(known)
    assert np.isclose(threshold, 94.05)
    metrics = known_unknown_metrics(known, np.array([100.0, 101.0]), threshold)
    assert metrics["auroc"] == 1.0
    assert metrics["unknown_rejection_rate"] == 1.0


def test_proser_bias_and_score_are_validation_only_calibratable() -> None:
    known = torch.tensor([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    dummy = torch.tensor([[1.0], [2.0], [3.0]])
    bias = calibrate_dummy_bias(known, dummy)
    scores = placeholder_unknownness(known, dummy, bias)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_evaluation_module_imports_unknowns_only_inside_main() -> None:
    source = (ROOT / "evaluate_osr.py").read_text()
    prefix = source.split("def main()", maxsplit=1)[0]
    assert "cifar100_unknowns" not in prefix


def test_no_rpl_implementation_or_configuration_exists() -> None:
    assert not list(ROOT.rglob("*rpl*"))
