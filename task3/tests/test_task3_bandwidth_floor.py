"""Target-free tests for the supplementary bandwidth-floor study."""

from __future__ import annotations

import unittest

import torch

from shared.mmd import l2_normalize_alignment_features, three_kernel_mmd
from task3.research_variants.bandwidth_floor import (
    PAIR_NAMES,
    dan_dg_pairwise_mmd_with_floor,
    reduce_calibration_medians,
    three_kernel_mmd_with_floor,
)
from task3.research_variants.config import (
    VARIANT_LOCKED,
    VARIANT_RUNS,
    load_variant_config,
)


class BandwidthFloorConfigurationTests(unittest.TestCase):
    def test_three_variant_lambdas_share_every_nonlambda_setting(self) -> None:
        configs = {run_id: load_variant_config(run_id) for run_id in VARIANT_RUNS}
        self.assertEqual(
            {config["mmd_lambda"] for config in configs.values()}, {0.1, 1.0, 10.0}
        )
        for run_id, config in configs.items():
            self.assertEqual(config["run_id"], run_id)
            self.assertEqual(config["method"], "dan_dg_bandwidth_floor")
            for name, expected in VARIANT_LOCKED.items():
                self.assertEqual(config[name], expected)

    def test_variant_is_never_marked_as_primary_replacement(self) -> None:
        self.assertFalse(VARIANT_LOCKED["primary_protocol_replacement"])
        self.assertFalse(VARIANT_LOCKED["target_domain_used"])
        self.assertEqual(VARIANT_LOCKED["floor_multiplier"], 1.0)


class BandwidthFloorMathematicsTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(6304)
        self.source = l2_normalize_alignment_features(
            torch.randn(8, 32, generator=generator)
        )
        self.target = l2_normalize_alignment_features(
            torch.randn(8, 32, generator=generator)
        )

    def test_inactive_floor_is_exactly_the_locked_mmd(self) -> None:
        expected, locked = three_kernel_mmd(self.source, self.target)
        actual, details = three_kernel_mmd_with_floor(
            self.source,
            self.target,
            bandwidth_floor=locked.median_squared_distance / 2.0,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(details.floor_activated)
        self.assertEqual(
            details.current_median_squared_distance,
            locked.median_squared_distance,
        )
        self.assertEqual(
            details.effective_median_squared_distance,
            locked.median_squared_distance,
        )

    def test_active_floor_changes_only_the_effective_median(self) -> None:
        _expected, locked = three_kernel_mmd(self.source, self.target)
        floor = locked.median_squared_distance * 2.0
        value, details = three_kernel_mmd_with_floor(
            self.source,
            self.target,
            bandwidth_floor=floor,
        )
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(details.floor_activated)
        self.assertEqual(details.frozen_floor_squared_distance, floor)
        self.assertEqual(details.effective_median_squared_distance, floor)
        self.assertEqual(
            details.bandwidths,
            (0.5 * floor, floor, 2.0 * floor),
        )

    def test_pairwise_floor_objective_has_finite_gradients(self) -> None:
        generator = torch.Generator().manual_seed(6304)
        features = torch.randn(24, 512, generator=generator, requires_grad=True)
        floors = {pair: 0.2 for pair in PAIR_NAMES}
        value, diagnostics = dan_dg_pairwise_mmd_with_floor(features, floors)
        self.assertTrue(torch.isfinite(value))
        self.assertEqual([item.pair for item in diagnostics], list(PAIR_NAMES))
        value.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_floor_stops_instead_of_silently_clamping(self) -> None:
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    three_kernel_mmd_with_floor(
                        self.source, self.target, bandwidth_floor=value
                    )


class BandwidthFloorCalibrationTests(unittest.TestCase):
    def test_reduction_uses_the_middle_of_exactly_235_values(self) -> None:
        samples = [float(index + 1) for index in range(235)]
        values = {
            pair: list(reversed(samples)) if index % 2 else list(samples)
            for index, pair in enumerate(PAIR_NAMES)
        }
        reduced = reduce_calibration_medians(values)
        for pair in PAIR_NAMES:
            self.assertEqual(reduced[pair]["sample_count"], 235)
            self.assertEqual(reduced[pair]["median"], 118.0)
            self.assertEqual(reduced[pair]["floor_squared_distance"], 118.0)
            self.assertEqual(reduced[pair]["floor_multiplier"], 1.0)


if __name__ == "__main__":
    unittest.main()
