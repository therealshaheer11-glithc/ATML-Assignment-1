
"""Frozen Task 1 backbone wrappers with model-specific normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import open_clip
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OPENAI_CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

SUPPORTED_BACKBONES = (
    "resnet50",
    "vit_b_16",
    "clip_vit_b_32",
)


@dataclass(frozen=True)
class BackboneMetadata:
    key: str
    display_name: str
    weights_identifier: str
    representation: str
    feature_dim: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]


class FrozenBackbone(nn.Module):
    """Base wrapper accepting common RGB tensors in the [0, 1] range."""

    def __init__(
        self,
        encoder: nn.Module,
        metadata: BackboneMetadata,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.metadata = metadata

        self.register_buffer(
            "_normalization_mean",
            torch.tensor(metadata.normalization_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_normalization_std",
            torch.tensor(metadata.normalization_std).view(1, 3, 1, 1),
            persistent=False,
        )

        self.requires_grad_(False)
        self.eval()

    @property
    def feature_dim(self) -> int:
        return self.metadata.feature_dim

    @property
    def device(self) -> torch.device:
        return self._normalization_mean.device

    def train(self, mode: bool = True) -> "FrozenBackbone":
        """Keep the pretrained backbone in evaluation mode permanently."""
        super().train(False)
        return self

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"Expected a 4D [batch, channels, height, width] tensor; "
                f"received shape {tuple(images.shape)}."
            )

        if tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError(
                "Every backbone must receive the same 224x224 RGB image. "
                f"Received shape {tuple(images.shape[1:])}."
            )

        if not images.is_floating_point():
            raise TypeError(
                "Images must be floating-point tensors scaled to [0, 1]."
            )

    def _normalize(self, images: Tensor) -> Tensor:
        return (
            images - self._normalization_mean
        ) / self._normalization_std

    def _encode_normalized(self, normalized_images: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, images: Tensor) -> Tensor:
        self._validate_images(images)
        normalized_images = self._normalize(images)
        features = self._encode_normalized(normalized_images)

        if features.ndim != 2:
            raise RuntimeError(
                f"Expected 2D feature matrix; received {tuple(features.shape)}."
            )

        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Expected feature dimension {self.feature_dim}; "
                f"received {features.shape[1]}."
            )

        return features


class ResNet50Backbone(FrozenBackbone):
    """ImageNet-V2 ResNet-50 global-average-pooled representation."""

    def __init__(self) -> None:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()

        metadata = BackboneMetadata(
            key="resnet50",
            display_name="ResNet-50",
            weights_identifier="ResNet50_Weights.IMAGENET1K_V2",
            representation="global_average_pooled_feature",
            feature_dim=feature_dim,
            normalization_mean=IMAGENET_MEAN,
            normalization_std=IMAGENET_STD,
        )

        super().__init__(encoder=model, metadata=metadata)

    def _encode_normalized(self, normalized_images: Tensor) -> Tensor:
        return self.encoder(normalized_images)


class ViTB16Backbone(FrozenBackbone):
    """ImageNet-V1 ViT-B/16 final class-token representation."""

    def __init__(self) -> None:
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        feature_dim = model.hidden_dim

        # Torchvision's forward pass sends the final class token through
        # `heads`. Replacing that module with Identity exposes the token.
        model.heads = nn.Identity()

        metadata = BackboneMetadata(
            key="vit_b_16",
            display_name="ViT-B/16",
            weights_identifier="ViT_B_16_Weights.IMAGENET1K_V1",
            representation="final_class_token",
            feature_dim=feature_dim,
            normalization_mean=IMAGENET_MEAN,
            normalization_std=IMAGENET_STD,
        )

        super().__init__(encoder=model, metadata=metadata)

    def _encode_normalized(self, normalized_images: Tensor) -> Tensor:
        return self.encoder(normalized_images)


class CLIPViTB32Backbone(FrozenBackbone):
    """OpenAI OpenCLIP ViT-B/32 normalized image representation."""

    model_name = "ViT-B-32"
    pretrained = "openai"

    def __init__(self) -> None:
        model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            force_quick_gelu=True,
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        feature_dim = int(model.visual.output_dim)

        metadata = BackboneMetadata(
            key="clip_vit_b_32",
            display_name="OpenCLIP ViT-B/32",
            weights_identifier="ViT-B-32 pretrained=openai",
            representation="l2_normalized_image_embedding",
            feature_dim=feature_dim,
            normalization_mean=OPENAI_CLIP_MEAN,
            normalization_std=OPENAI_CLIP_STD,
        )

        super().__init__(encoder=model, metadata=metadata)
        self.tokenizer = tokenizer

    def _encode_normalized(self, normalized_images: Tensor) -> Tensor:
        features = self.encoder.encode_image(normalized_images)
        return F.normalize(features, dim=-1)

    @torch.inference_mode()
    def encode_text(self, prompts: Sequence[str]) -> Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        features = self.encoder.encode_text(tokens)
        return F.normalize(features, dim=-1)

    @torch.inference_mode()
    def zero_shot_logits(
        self,
        images: Tensor,
        class_names: Sequence[str],
        prompt_template: str = "a photo of a {class}.",
    ) -> Tensor:
        prompts = [
            prompt_template.format_map({"class": class_name})
            for class_name in class_names
        ]

        image_features = self(images)
        text_features = self.encode_text(prompts)

        # Use CLIP's learned similarity scale as required.
        scale = self.encoder.logit_scale.exp()
        return scale * image_features @ text_features.T


def build_backbone(
    name: str,
    device: str | torch.device | None = None,
) -> FrozenBackbone:
    """Construct one required frozen backbone without loading the others."""

    if name == "resnet50":
        backbone: FrozenBackbone = ResNet50Backbone()
    elif name == "vit_b_16":
        backbone = ViTB16Backbone()
    elif name == "clip_vit_b_32":
        backbone = CLIPViTB32Backbone()
    else:
        raise ValueError(
            f"Unsupported backbone {name!r}. "
            f"Choose one of {SUPPORTED_BACKBONES}."
        )

    selected_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    return backbone.to(selected_device).eval()
