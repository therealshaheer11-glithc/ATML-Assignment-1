"""Fast, target-free tests for the approved Task 3 implementation."""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from shared.mmd import l2_normalize_alignment_features, three_kernel_mmd
from task3.config import LOCKED_BASE, RUNS, load_config
from task3.data import CyclingBatchSampler, image_transform, validate_source_record
from task3.methods.dan_dg import dan_dg_pairwise_mmd
from task3.methods.sam import clip_gradients, sam_step
from task3.model import (
    assert_batchnorm_unchanged,
    batchnorm_buffers,
    freeze_batchnorm_statistics,
)


class ConfigurationTests(unittest.TestCase):
    def test_every_approved_run_loads_with_the_locked_base(self) -> None:
        for run_id, (method, mmd_lambda, sam_rho) in RUNS.items():
            with self.subTest(run_id=run_id):
                config = load_config(run_id)
                for name, value in LOCKED_BASE.items():
                    self.assertEqual(config[name], value)
                self.assertEqual(config["method"], method)
                self.assertEqual(config["mmd_lambda"], mmd_lambda)
                self.assertEqual(config["sam_rho"], sam_rho)

    def test_normalization_policies_are_not_conflated(self) -> None:
        config = load_config("dan_dg_1")
        self.assertEqual(
            config["mmd_feature_normalization"],
            "l2_per_sample_mmd_input_only",
        )
        self.assertEqual(
            config["adversarial_feature_normalization"],
            "not_applicable_to_task3",
        )


class SourceEmbargoAndSamplingTests(unittest.TestCase):
    def test_source_record_rejects_sketch(self) -> None:
        with self.assertRaises((ValueError, RuntimeError)):
            validate_source_record(
                {"path": "sketch/dog/example.jpg", "class_id": 0},
                "photo",
            )

    def test_source_record_rejects_nested_sketch_component(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_source_record(
                {"path": "photo/sketch/example.jpg", "class_id": 0},
                "photo",
            )

    def test_cycling_sampler_is_seeded_and_complete(self) -> None:
        first = list(CyclingBatchSampler(count=5, batch_size=3, steps=4, seed=6304))
        second = list(CyclingBatchSampler(count=5, batch_size=3, steps=4, seed=6304))
        third = list(CyclingBatchSampler(count=5, batch_size=3, steps=4, seed=6305))
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual([len(batch) for batch in first], [3, 3, 3, 3])
        self.assertTrue(all(0 <= index < 5 for batch in first for index in batch))

    def test_transforms_match_the_locked_task2_pipeline(self) -> None:
        train_names = [type(operation).__name__ for operation in image_transform(True).transforms]
        validation_names = [
            type(operation).__name__ for operation in image_transform(False).transforms
        ]
        self.assertEqual(
            train_names,
            ["Resize", "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"],
        )
        self.assertEqual(
            validation_names,
            ["Resize", "CenterCrop", "ToTensor", "Normalize"],
        )


class BatchNormTests(unittest.TestCase):
    def test_running_statistics_freeze_but_affine_parameters_train(self) -> None:
        model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
        model.train()
        expected = batchnorm_buffers(model)
        freeze_batchnorm_statistics(model)

        batchnorm = model[0]
        self.assertFalse(batchnorm.training)
        self.assertTrue(batchnorm.weight.requires_grad)
        self.assertTrue(batchnorm.bias.requires_grad)

        loss = model(torch.randn(8, 3)).square().mean()
        loss.backward()
        self.assertIsNotNone(batchnorm.weight.grad)
        self.assertIsNotNone(batchnorm.bias.grad)
        assert_batchnorm_unchanged(model, expected)


class DAN_DGTests(unittest.TestCase):
    def test_pairwise_mmd_is_the_mean_of_three_locked_mmd_values(self) -> None:
        generator = torch.Generator().manual_seed(6304)
        features = torch.randn(24, 512, generator=generator, requires_grad=True)
        original = features.detach().clone()

        actual, diagnostics = dan_dg_pairwise_mmd(features, batch_per_domain=8)
        domains = [
            l2_normalize_alignment_features(block)
            for block in features.split(8, dim=0)
        ]
        expected_values = [
            three_kernel_mmd(domains[0], domains[1])[0],
            three_kernel_mmd(domains[0], domains[2])[0],
            three_kernel_mmd(domains[1], domains[2])[0],
        ]
        expected = torch.stack(expected_values).mean()

        self.assertTrue(torch.equal(features.detach(), original))
        self.assertTrue(torch.allclose(actual, expected, atol=0.0, rtol=0.0))
        self.assertEqual(
            [item.pair for item in diagnostics],
            [
                "photo__art_painting",
                "photo__cartoon",
                "art_painting__cartoon",
            ],
        )
        self.assertTrue(all(item.distinct_pair_count == 120 for item in diagnostics))
        for item in diagnostics:
            self.assertEqual(
                item.bandwidths,
                (
                    0.5 * item.median_squared_distance,
                    item.median_squared_distance,
                    2.0 * item.median_squared_distance,
                ),
            )

        actual.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_off_diagonal_zero_distances_are_retained(self) -> None:
        base = torch.eye(8)
        features = torch.cat((base, base.clone(), torch.roll(base, shifts=1, dims=1)))
        _value, diagnostics = dan_dg_pairwise_mmd(features, batch_per_domain=8)
        self.assertGreater(diagnostics[0].off_diagonal_zero_count, 0)
        self.assertGreater(diagnostics[1].off_diagonal_zero_count, 0)

    def test_nonpositive_median_stops_instead_of_being_clamped(self) -> None:
        features = torch.ones(24, 512)
        with self.assertRaises(FloatingPointError):
            dan_dg_pairwise_mmd(features, batch_per_domain=8)


class SAMTests(unittest.TestCase):
    def test_standard_sam_restores_then_steps_from_original_parameters(self) -> None:
        parameter = nn.Parameter(torch.tensor([1.0]))
        model = nn.ParameterList([parameter])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def closure() -> torch.Tensor:
            return 0.5 * parameter.square().sum()

        diagnostics = sam_step(
            model,
            optimizer,
            closure,
            rho=0.05,
            max_gradient_norm=20.0,
        )

        self.assertAlmostEqual(diagnostics.first_gradient_norm, 1.0, places=6)
        self.assertAlmostEqual(diagnostics.perturbation_norm, 0.05, places=6)
        self.assertAlmostEqual(diagnostics.second_gradient_norm, 1.05, places=6)
        self.assertAlmostEqual(parameter.item(), 0.895, places=6)
        self.assertFalse(diagnostics.second_gradient_was_clipped)

    def test_only_the_second_sam_gradient_is_clipped(self) -> None:
        parameter = nn.Parameter(torch.tensor([100.0]))
        model = nn.ParameterList([parameter])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        diagnostics = sam_step(
            model,
            optimizer,
            lambda: 0.5 * parameter.square().sum(),
            rho=0.05,
            max_gradient_norm=2.0,
        )

        self.assertAlmostEqual(diagnostics.first_gradient_norm, 100.0, places=4)
        self.assertAlmostEqual(diagnostics.perturbation_norm, 0.05, places=5)
        self.assertGreater(diagnostics.second_gradient_norm, 100.0)
        self.assertLessEqual(diagnostics.second_gradient_norm_after_clipping, 2.00002)
        self.assertTrue(diagnostics.second_gradient_was_clipped)
        self.assertAlmostEqual(parameter.item(), 99.8, places=4)

    def test_perturbed_pass_failure_restores_parameters(self) -> None:
        parameter = nn.Parameter(torch.tensor([2.0]))
        model = nn.ParameterList([parameter])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        calls = 0

        def closure() -> torch.Tensor:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("deliberate second-pass failure")
            return parameter.square().sum()

        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            sam_step(
                model,
                optimizer,
                closure,
                rho=0.05,
                max_gradient_norm=20.0,
            )
        self.assertEqual(parameter.item(), 2.0)

    def test_global_gradient_clipping_uses_l2_norm(self) -> None:
        left = nn.Parameter(torch.tensor([0.0]))
        right = nn.Parameter(torch.tensor([0.0]))
        left.grad = torch.tensor([30.0])
        right.grad = torch.tensor([40.0])
        pre, post, clipped = clip_gradients([left, right], max_norm=20.0)
        self.assertAlmostEqual(pre, 50.0, places=6)
        self.assertAlmostEqual(post, 20.0, places=4)
        self.assertTrue(clipped)
        self.assertAlmostEqual(math.hypot(left.grad.item(), right.grad.item()), 20.0, places=4)


if __name__ == "__main__":
    unittest.main()
