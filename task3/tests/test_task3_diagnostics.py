"""Fast target-free tests for the approved Task 3 source diagnostics."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from task3.data import SOURCES
from task3.evaluation.sharpness import (
    common_sharpness_proxy,
    select_sharpness_records,
)
from task3.evaluation.source_domain_separability import (
    fit_domain_probe,
    make_probe_partition,
    select_probe_records,
)


def synthetic_protocol(count: int) -> dict:
    return {
        "source_splits": {
            domain: {
                "validation": [
                    {
                        "path": f"{domain}/class_{index % 7}/image_{index:03d}.jpg",
                        "class_id": index % 7,
                    }
                    for index in range(count)
                ]
            }
            for domain in SOURCES
        }
    }


class SourceDomainProbeTests(unittest.TestCase):
    def test_probe_selection_is_balanced_seeded_and_target_free(self) -> None:
        protocol = synthetic_protocol(12)
        first = select_probe_records(protocol, per_domain=10)
        second = select_probe_records(protocol, per_domain=10)
        self.assertEqual(first, second)
        self.assertEqual({domain: len(first[domain]) for domain in SOURCES}, {
            domain: 10 for domain in SOURCES
        })
        self.assertTrue(
            all(
                record["path"].startswith(domain + "/")
                for domain in SOURCES
                for record in first[domain]
            )
        )

    def test_probe_partition_is_shared_disjoint_and_domain_stratified(self) -> None:
        selected = select_probe_records(synthetic_protocol(12), per_domain=10)
        partition = make_probe_partition(selected)
        train = set(partition["train_indices"])
        test = set(partition["test_indices"])
        self.assertFalse(train.intersection(test))
        self.assertEqual(train.union(test), set(range(30)))
        labels = np.asarray(partition["domain_labels"])
        test_counts = np.bincount(labels[list(test)], minlength=3)
        self.assertTrue(np.all(test_counts == 3))

    def test_probe_fits_training_only_scaled_multinomial_classifier(self) -> None:
        selected = select_probe_records(synthetic_protocol(14), per_domain=12)
        partition = make_probe_partition(selected)
        labels = np.asarray(partition["domain_labels"])
        features = np.zeros((len(labels), 512), dtype=np.float64)
        features[np.arange(len(labels)), labels] = 10.0
        result = fit_domain_probe(features, partition)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["train_count"] + result["test_count"], len(labels))
        self.assertEqual(
            result["standard_scaler_fit_scope"], "probe_training_features_only"
        )


class _TupleLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(4, 3)

    def forward(self, inputs: torch.Tensor):
        return self.classifier(inputs), inputs


class SharpnessDiagnosticTests(unittest.TestCase):
    def test_selection_is_deterministic_and_sharpness_restores_exactly(self) -> None:
        first = select_sharpness_records(synthetic_protocol(40))
        second = select_sharpness_records(synthetic_protocol(40))
        self.assertEqual(first, second)
        self.assertEqual([len(first[domain]) for domain in SOURCES], [32, 32, 32])

        torch.manual_seed(6304)
        model = _TupleLinear()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        images = torch.randn(12, 4)
        labels = torch.arange(12) % 3
        result = common_sharpness_proxy(model, images, labels, radius=0.05)
        self.assertAlmostEqual(result["actual_perturbation_norm"], 0.05, places=6)
        self.assertGreater(result["gradient_norm"], 0.0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))


if __name__ == "__main__":
    unittest.main()
