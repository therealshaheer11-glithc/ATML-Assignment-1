from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from task4.models import CifarResNet18, ProserResNet18


@dataclass(frozen=True)
class ModelOutputs:
    known_logits: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    dummy_logits: np.ndarray | None = None
    original_indices: np.ndarray | None = None


def load_model_from_checkpoint(path: str, method: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != method:
        raise ValueError(f"Checkpoint method mismatch: expected {method}")
    if method == "proser":
        model = ProserResNet18(CifarResNet18(), dummy_classifiers=5)
    else:
        model = CifarResNet18()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def extract_outputs(model, loader: DataLoader, device: torch.device) -> ModelOutputs:
    known_parts = []
    dummy_parts = []
    feature_parts = []
    label_parts = []
    index_parts = []
    running_index = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            labels = torch.as_tensor(batch[1])
            if isinstance(model, ProserResNet18):
                known, dummy, features = model(images, return_features=True)
                dummy_parts.append(dummy.cpu())
            else:
                known, features = model(images, return_features=True)
            known_parts.append(known.cpu())
            feature_parts.append(features.cpu())
            label_parts.append(labels.cpu())
            if len(batch) >= 3:
                index_parts.append(torch.as_tensor(batch[2]).cpu())
            else:
                index_parts.append(torch.arange(running_index, running_index + len(labels)))
            running_index += len(labels)
    return ModelOutputs(
        known_logits=torch.cat(known_parts).numpy(),
        dummy_logits=torch.cat(dummy_parts).numpy() if dummy_parts else None,
        features=torch.cat(feature_parts).numpy(),
        labels=torch.cat(label_parts).numpy(),
        original_indices=torch.cat(index_parts).numpy(),
    )

