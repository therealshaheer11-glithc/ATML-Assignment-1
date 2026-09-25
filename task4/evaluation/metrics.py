from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def validation_threshold(scores: np.ndarray, quantile: float = 0.95) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Validation scores must be finite and one-dimensional")
    return float(np.quantile(scores, quantile, method="linear"))


def known_unknown_metrics(
    known_scores: np.ndarray, unknown_scores: np.ndarray, threshold: float
) -> dict[str, float]:
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    labels = np.concatenate(
        [np.zeros(len(known_scores), dtype=np.int64), np.ones(len(unknown_scores), dtype=np.int64)]
    )
    scores = np.concatenate([known_scores, unknown_scores])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "known_acceptance_rate": float(np.mean(known_scores <= threshold)),
        "unknown_rejection_rate": float(np.mean(unknown_scores > threshold)),
        "fpr_at_95_tpr": float(np.mean(unknown_scores <= threshold)),
        "threshold": float(threshold),
    }

