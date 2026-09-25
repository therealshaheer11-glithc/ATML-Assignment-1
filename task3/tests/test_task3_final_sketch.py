"""Target-free unit tests for the locked final-evaluation calculations."""

from __future__ import annotations

import unittest

import numpy as np

from task3.data import CLASSES
from task3.evaluation.final_sketch import (
    metrics_from_predictions,
    select_change_examples,
    target_class_id,
    validate_target_records,
)


class FinalSketchCalculationTests(unittest.TestCase):
    def test_target_class_parser_rejects_non_sketch(self) -> None:
        with self.assertRaises(ValueError):
            target_class_id("photo/dog/example.jpg")

    def test_target_class_parser_uses_parent_folder(self) -> None:
        self.assertEqual(target_class_id("sketch/guitar/example.jpg"), 3)

    def test_record_validation_requires_complete_locked_count(self) -> None:
        with self.assertRaises(ValueError):
            validate_target_records([{"path": "sketch/dog/a.jpg", "id": "a"}])

    def test_metrics_fix_macro_f1_to_all_classes(self) -> None:
        truth = np.arange(len(CLASSES), dtype=np.int64)
        predicted = truth.copy()
        result = metrics_from_predictions(truth, predicted)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["confusion_matrix"], np.eye(7, dtype=int).tolist())

    def test_change_examples_use_fixed_class_and_path_ties(self) -> None:
        records = [
            {"id": "b", "path": "sketch/dog/b.jpg"},
            {"id": "a", "path": "sketch/dog/a.jpg"},
            {"id": "c", "path": "sketch/elephant/c.jpg"},
            {"id": "d", "path": "sketch/elephant/d.jpg"},
        ]
        truth = np.asarray([0, 0, 1, 1])
        erm = np.asarray([1, 0, 1, 0])
        model = np.asarray([0, 1, 0, 1])
        confidence = np.asarray([0.9, 0.8, 0.7, 0.6])
        erm_metrics = metrics_from_predictions(truth, erm)
        model_metrics = metrics_from_predictions(truth, model)
        summary, examples = select_change_examples(
            "candidate",
            records,
            truth,
            erm,
            model,
            confidence,
            confidence,
            erm_metrics,
            model_metrics,
        )
        self.assertEqual(summary["largest_improvement_class"], "dog")
        self.assertEqual(summary["largest_degradation_class"], "dog")
        self.assertEqual(examples[0]["path"], "sketch/dog/b.jpg")
        self.assertEqual(examples[1]["path"], "sketch/dog/a.jpg")


if __name__ == "__main__":
    unittest.main()
