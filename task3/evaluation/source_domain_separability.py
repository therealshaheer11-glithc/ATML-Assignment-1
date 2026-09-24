"""Deterministic source-domain separability diagnostic for Task 3."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from task3.data import SEED, SOURCES


PROBE_PER_DOMAIN = 334
PROBE_TEST_FRACTION = 0.30
PROBE_C = 1.0
PROBE_SOLVER = "lbfgs"
PROBE_MAX_ITER = 2000


def _seeded_subset(
    records: Sequence[dict],
    count: int,
    generator: np.random.Generator,
) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Cannot select {count} records from only {len(records)}")
    if len(records) == count:
        return [dict(record) for record in records]
    indices = np.sort(generator.choice(len(records), size=count, replace=False))
    return [dict(records[int(index)]) for index in indices]


def select_probe_records(
    protocol: dict,
    *,
    seed: int = SEED,
    per_domain: int = PROBE_PER_DOMAIN,
) -> dict[str, list[dict]]:
    """Select one domain-balanced validation subset with a single seeded RNG stream."""
    generator = np.random.default_rng(seed)
    selected: dict[str, list[dict]] = {}
    for domain in SOURCES:
        records = protocol["source_splits"][domain]["validation"]
        selected[domain] = _seeded_subset(records, per_domain, generator)
    return selected


def make_probe_partition(
    selected: dict[str, list[dict]],
    *,
    seed: int = SEED,
    test_fraction: float = PROBE_TEST_FRACTION,
) -> dict:
    identifiers: list[str] = []
    domain_labels: list[int] = []
    domains: list[str] = []
    class_ids: list[int] = []

    for domain_id, domain in enumerate(SOURCES):
        for record in selected[domain]:
            identifiers.append(str(record["path"]))
            domain_labels.append(domain_id)
            domains.append(domain)
            class_ids.append(int(record["class_id"]))

    all_indices = np.arange(len(identifiers), dtype=np.int64)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=np.asarray(domain_labels, dtype=np.int64),
    )

    return {
        "seed": seed,
        "per_domain": len(selected[SOURCES[0]]),
        "test_fraction": test_fraction,
        "identifiers": identifiers,
        "domains": domains,
        "domain_labels": domain_labels,
        "class_ids": class_ids,
        "train_indices": [int(value) for value in train_indices],
        "test_indices": [int(value) for value in test_indices],
    }


def fit_domain_probe(features: np.ndarray, partition: dict) -> dict:
    """Fit the locked training-only-scaled multinomial domain classifier."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError(f"Expected an N x 512 feature matrix, got {features.shape}")
    labels = np.asarray(partition["domain_labels"], dtype=np.int64)
    if len(features) != len(labels):
        raise ValueError("Feature and domain-label counts differ")

    train_indices = np.asarray(partition["train_indices"], dtype=np.int64)
    test_indices = np.asarray(partition["test_indices"], dtype=np.int64)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_indices])
    test_features = scaler.transform(features[test_indices])

    classifier = LogisticRegression(
        C=PROBE_C,
        solver=PROBE_SOLVER,
        max_iter=PROBE_MAX_ITER,
        random_state=SEED,
    )
    classifier.fit(train_features, labels[train_indices])
    predictions = classifier.predict(test_features)
    accuracy = float(accuracy_score(labels[test_indices], predictions))
    matrix = confusion_matrix(
        labels[test_indices], predictions, labels=list(range(len(SOURCES)))
    )
    if not np.isfinite(accuracy):
        raise FloatingPointError("Source-domain probe produced non-finite accuracy")
    if int(np.max(classifier.n_iter_)) >= PROBE_MAX_ITER:
        raise RuntimeError("Source-domain probe reached its iteration limit")

    return {
        "accuracy": accuracy,
        "chance_accuracy": 1.0 / len(SOURCES),
        "confusion_matrix": matrix.astype(int).tolist(),
        "train_count": int(len(train_indices)),
        "test_count": int(len(test_indices)),
        "standard_scaler_fit_scope": "probe_training_features_only",
        "logistic_regression": {
            "C": PROBE_C,
            "solver": PROBE_SOLVER,
            "max_iter": PROBE_MAX_ITER,
            "random_state": SEED,
            "multi_class": "multinomial_automatic_for_three_classes_with_lbfgs",
            "iterations": [int(value) for value in classifier.n_iter_],
        },
    }
