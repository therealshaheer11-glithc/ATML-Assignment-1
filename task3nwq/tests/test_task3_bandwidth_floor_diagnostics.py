"""Target-free tests for the bandwidth-floor source-diagnostic audit."""

from __future__ import annotations

import copy
import unittest

from task3.data import SEED, SOURCES
from task3.evaluation.sharpness import SHARPNESS_RADIUS, select_sharpness_records
from task3.evaluation.source_domain_separability import (
    make_probe_partition,
    select_probe_records,
)
from task3.research_variants.config import VARIANT_ID, VARIANT_PROTOCOL_VERSION
from task3.research_variants.run_source_diagnostics import (
    MODEL_SPECS,
    assert_checkpoint_metadata,
    assert_probe_design,
    assert_sharpness_design,
)


def synthetic_protocol(count: int = 40) -> dict:
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


class LockedDiagnosticDesignTests(unittest.TestCase):
    def test_probe_requires_the_exact_reconstructed_partition(self) -> None:
        selected = select_probe_records(synthetic_protocol(), per_domain=32)
        partition = make_probe_partition(selected)
        saved = {
            "status": "TASK3_SOURCE_DOMAIN_PROBE_DESIGN_LOCKED",
            "source_only_phase": True,
            "sketch_images_accessed": 0,
            "selection_rng": (
                "numpy.default_rng_PCG64_single_stream_domain_order_P_A_C"
            ),
            "feature_representation": "raw_unnormalized_512d_pre_classifier",
            "partition": partition,
        }
        assert_probe_design(saved, partition)
        changed = copy.deepcopy(saved)
        changed["partition"]["identifiers"][0] = "sketch/forbidden.jpg"
        with self.assertRaises(RuntimeError):
            assert_probe_design(changed, partition)

    def test_sharpness_requires_the_exact_independent_selection(self) -> None:
        selected = select_sharpness_records(synthetic_protocol())
        saved = {
            "status": "TASK3_SHARPNESS_BATCH_DESIGN_LOCKED",
            "source_only_phase": True,
            "sketch_images_accessed": 0,
            "selection_rng": (
                "numpy.default_rng_PCG64_independent_stream_domain_order_P_A_C"
            ),
            "seed": SEED,
            "per_domain": 32,
            "radius": SHARPNESS_RADIUS,
            "records": selected,
        }
        assert_sharpness_design(saved, selected)
        changed = copy.deepcopy(saved)
        changed["radius"] = 0.1
        with self.assertRaises(RuntimeError):
            assert_sharpness_design(changed, selected)


class VariantCheckpointGateTests(unittest.TestCase):
    def checkpoint_payload(self, run_id: str) -> dict:
        spec = MODEL_SPECS[run_id]
        payload = {
            "source_only_phase": True,
            "sketch_images_accessed": 0,
            "epoch": spec["epoch"],
            "identity": {
                "code_sha256": spec["training_code_sha256"],
                "config": {
                    "run_id": run_id,
                    "method": spec["method"],
                    "mmd_lambda": spec["mmd_lambda"],
                },
            },
        }
        if spec["research_variant"]:
            payload.update(
                {
                    "research_variant": True,
                    "primary_protocol_replacement": False,
                    "final_evaluation_started": False,
                }
            )
            payload["identity"].update(
                {
                    "variant_id": VARIANT_ID,
                    "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
                }
            )
        return payload

    def test_all_five_pending_checkpoint_identities_pass(self) -> None:
        for run_id in MODEL_SPECS:
            with self.subTest(run_id=run_id):
                assert_checkpoint_metadata(self.checkpoint_payload(run_id), run_id)

    def test_target_access_or_training_code_change_is_rejected(self) -> None:
        payload = self.checkpoint_payload("dan_dg_floor_1")
        payload["sketch_images_accessed"] = 1
        with self.assertRaises(RuntimeError):
            assert_checkpoint_metadata(payload, "dan_dg_floor_1")

        payload = self.checkpoint_payload("dan_dg_floor_1")
        payload["identity"]["code_sha256"] = "changed"
        with self.assertRaises(RuntimeError):
            assert_checkpoint_metadata(payload, "dan_dg_floor_1")


if __name__ == "__main__":
    unittest.main()
