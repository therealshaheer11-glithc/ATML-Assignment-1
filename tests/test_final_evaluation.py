"""Focused tests for the post-freeze Task 2 evaluator."""

from __future__ import annotations

import numpy as np

from task2.evaluate_frozen import (
    CLASSES,
    classification_metrics,
    dominant_confusion_rows,
    make_probe_partition,
    target_class_id,
)


def test_target_label_is_parsed_only_from_expected_sketch_path() -> None:
    assert target_class_id("sketch/dog/example.png") == CLASSES.index("dog")
    try:
        target_class_id("photo/dog/example.png")
    except ValueError:
        pass
    else:
        raise AssertionError("A non-Sketch path must fail")


def test_macro_f1_always_includes_all_seven_classes() -> None:
    truth = np.arange(7)
    predicted = np.zeros(7, dtype=int)
    metrics = classification_metrics(truth, predicted)
    expected_class_zero_f1 = 2.0 / 8.0
    assert np.isclose(metrics["macro_f1"], expected_class_zero_f1 / 7.0)


def test_probe_partition_is_balanced_shared_and_reproducible() -> None:
    first = make_probe_partition(11, 29)
    second = make_probe_partition(11, 29)
    assert first == second
    assert len(first["selected_target_indices"]) == 11
    assert len(set(first["selected_target_indices"])) == 11
    labels = np.asarray(first["domain_labels"])
    assert int((labels == 0).sum()) == int((labels == 1).sum()) == 11
    train = np.asarray(first["train_indices"])
    test = np.asarray(first["test_indices"])
    assert not set(train).intersection(set(test))
    assert set(train).union(set(test)) == set(range(22))


def test_dominant_confusion_excludes_correct_diagonal() -> None:
    matrix = np.eye(7, dtype=int) * 5
    matrix[0, 1] = 3
    matrix[0, 2] = 1
    rows = dominant_confusion_rows("example", matrix)
    assert rows[0]["dominant_wrong_prediction"] == CLASSES[1]
    assert rows[0]["dominant_confusion_count"] == 3
    assert rows[0]["total_class_errors"] == 4
    assert rows[1]["dominant_wrong_prediction"] == ""


def test_dominant_confusion_preserves_ties() -> None:
    matrix = np.eye(7, dtype=int)
    matrix[0, 1] = 3
    matrix[0, 2] = 3
    rows = dominant_confusion_rows("example", matrix)
    assert rows[0]["dominant_wrong_prediction"] == f"{CLASSES[1]}|{CLASSES[2]}"
