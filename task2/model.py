"""Shared ResNet-18 and the required frozen BatchNorm-statistics policy."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from shared.pacs import CLASSES


class PACSClassifier(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if backbone.fc.in_features != 512:
            raise RuntimeError("Expected a 512-dimensional ResNet-18 representation")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(512, len(CLASSES))

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(images)
        return self.classifier(features), features


def freeze_batchnorm_statistics(model: nn.Module) -> None:
    """Call after model.train(); gamma and beta remain trainable."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def batchnorm_buffers(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def assert_batchnorm_unchanged(model: nn.Module, original: dict[str, Tensor]) -> None:
    current = batchnorm_buffers(model)
    if current.keys() != original.keys() or any(not torch.equal(current[key], original[key]) for key in original):
        raise RuntimeError("BatchNorm running statistics changed from their pretrained values")
