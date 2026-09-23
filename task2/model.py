"""ResNet-18 classifier and the assignment's BatchNorm policy."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


NUM_CLASSES = 7
FEATURE_WIDTH = 512


class PACSClassifier(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        if backbone.fc.in_features != FEATURE_WIDTH:
            raise RuntimeError("Unexpected ResNet-18 feature width")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(FEATURE_WIDTH, NUM_CLASSES)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(images)
        return self.classifier(features), features


def freeze_batchnorm_statistics(model: nn.Module) -> None:
    """Freeze running buffers while leaving trainable gamma/beta parameters untouched."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def batchnorm_buffers(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def assert_batchnorm_unchanged(model: nn.Module, expected: dict[str, Tensor]) -> None:
    actual = batchnorm_buffers(model)
    if actual.keys() != expected.keys():
        raise RuntimeError("BatchNorm buffer keys changed")
    changed = [name for name in expected if not torch.equal(expected[name], actual[name])]
    if changed:
        raise RuntimeError(f"BatchNorm pretrained buffers changed: {changed[:3]}")


def state_dict_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

