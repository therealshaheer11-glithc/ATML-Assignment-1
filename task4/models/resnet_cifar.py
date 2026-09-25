from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class CifarResNet18(nn.Module):
    """Torchvision ResNet-18 with the PA-required CIFAR stem."""

    feature_dim = 512

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = resnet18(weights=None, num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        self.backbone.maxpool = nn.Identity()

    def forward_to_layer2(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        return model.layer2(x)

    def forward_from_layer2(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_layer2(self.forward_to_layer2(x))

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(features)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.classify_features(features)
        return (logits, features) if return_features else logits


class ProserResNet18(nn.Module):
    """CIFAR ResNet-18 with ten known and five PROSER dummy classifiers."""

    def __init__(self, known_model: CifarResNet18, dummy_classifiers: int = 5):
        super().__init__()
        if dummy_classifiers != 5:
            raise ValueError("The PA requires exactly five dummy classifiers")
        self.known_model = known_model
        self.dummy_classifier = nn.Linear(known_model.feature_dim, dummy_classifiers)

    @property
    def feature_dim(self) -> int:
        return self.known_model.feature_dim

    def forward_to_layer2(self, x: torch.Tensor) -> torch.Tensor:
        return self.known_model.forward_to_layer2(x)

    def forward_from_layer2(self, x: torch.Tensor) -> torch.Tensor:
        return self.known_model.forward_from_layer2(x)

    def classify_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.known_model.classify_features(features), self.dummy_classifier(features)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.known_model.forward_features(x)
        known_logits, dummy_logits = self.classify_features(features)
        if return_features:
            return known_logits, dummy_logits, features
        return known_logits, dummy_logits
