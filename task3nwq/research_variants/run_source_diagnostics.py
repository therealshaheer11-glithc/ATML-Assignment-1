"""Complete the locked source diagnostics on all still-pending checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from task3.config import code_tree_sha256
from task3.data import SEED, SOURCES, load_source_protocol, verify_source_snapshot
from task3.evaluation.run_source_diagnostics import (
    EXPECTED_RUNTIME,
    extract_probe_features,
    load_sharpness_batch,
    runtime_identity,
    seed_everything,
    validate_sources,
)
from task3.evaluation.sharpness import (
    SHARPNESS_RADIUS,
    common_sharpness_proxy,
    select_sharpness_records,
)
from task3.evaluation.source_domain_separability import (
    fit_domain_probe,
    make_probe_partition,
    select_probe_records,
)
from task3.model import PACSClassifier, state_dict_sha256
from task3.research_variants.config import VARIANT_ID, VARIANT_PROTOCOL_VERSION


TRAINING_COMMIT = "dc3acfdf547e7bc29bd381b3fe05e271879f18d0"
TRAINING_CODE_SHA256 = (
    "34d2778e76a5a7522a9c2eac2682eb4d2a09ee9de0e88b6d04d7d0a68e26ce13"
)
PRIMARY_TRAINING_COMMIT = "19208b4c62acb980fb3246f30e062784b90d8dfc"
PRIMARY_TRAINING_CODE_SHA256 = (
    "4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44"
)
PROBE_DESIGN_SHA256 = (
    "6f94666bd693328a6cdf6b2b1193a97267f7367b3ff62981c38a62651adc8f3c"
)
SHARPNESS_DESIGN_SHA256 = (
    "ac5ed81e14e7867dc616416de9da656e56d5c9fc88e02fe6ef66aa2ffffc7cab"
)

MODEL_ORDER = (
    "dan_dg_0p1",
    "dan_dg_10",
    "dan_dg_floor_1",
    "dan_dg_floor_0p1",
    "dan_dg_floor_10",
)

MODEL_SPECS = {
    "dan_dg_0p1": {
        "method": "dan_dg",
        "research_variant": False,
        "training_code_sha256": PRIMARY_TRAINING_CODE_SHA256,
        "mmd_lambda": 0.1,
        "file_sha256": (
            "dc6036a28e3af8c281b143adb6f47b4d03676fa44c117e912ef9bcd959d8ca27"
        ),
        "history_sha256": (
            "fc92efd81ea529a48373977f9d4973e754eb53cfca7622c8513ad5cfc64b35b8"
        ),
        "manifest_sha256": (
            "731f07bded36f6034e014eb217dca2b1b3a9c7836ed0e5bd72d0c2be3537672a"
        ),
        "epoch": 5,
        "source_validation": {
            "photo_accuracy": 0.9700598802395209,
            "photo_macro_f1": 0.9653057911899712,
            "art_painting_accuracy": 0.926829268292683,
            "art_painting_macro_f1": 0.9230060448010302,
            "cartoon_accuracy": 0.9466950959488273,
            "cartoon_macro_f1": 0.9503774930172837,
            "mean_source_accuracy": 0.9478614148270105,
            "mean_source_macro_f1": 0.9462297763360951,
            "worst_source_accuracy": 0.926829268292683,
            "worst_source_macro_f1": 0.9230060448010302,
        },
    },
    "dan_dg_10": {
        "method": "dan_dg",
        "research_variant": False,
        "training_code_sha256": PRIMARY_TRAINING_CODE_SHA256,
        "mmd_lambda": 10.0,
        "file_sha256": (
            "f8cc723dc16b5e17a48f8454541beb473b70e6c38d45895f170c09f59821b2c8"
        ),
        "history_sha256": (
            "dec8773f2a0642b2aecea85619dbd8bdc19a74723c06a5ce9ca24ce9c8b4b918"
        ),
        "manifest_sha256": (
            "962626d182952f068a689bb213474074f8f07869c9a5a254cc246164d8fe4330"
        ),
        "epoch": 7,
        "source_validation": {
            "photo_accuracy": 0.25748502994011974,
            "photo_macro_f1": 0.058503401360544216,
            "art_painting_accuracy": 0.21951219512195122,
            "art_painting_macro_f1": 0.05142857142857143,
            "cartoon_accuracy": 0.17270788912579957,
            "cartoon_macro_f1": 0.04207792207792208,
            "mean_source_accuracy": 0.21656837139595686,
            "mean_source_macro_f1": 0.05066996495567924,
            "worst_source_accuracy": 0.17270788912579957,
            "worst_source_macro_f1": 0.04207792207792208,
        },
    },
    "dan_dg_floor_1": {
        "method": "dan_dg_bandwidth_floor",
        "research_variant": True,
        "training_code_sha256": TRAINING_CODE_SHA256,
        "mmd_lambda": 1.0,
        "file_sha256": (
            "6b8c8ac0f373882b468e2fbf98bfeb4f97e1a799bf073cc6a7f3667d9747418a"
        ),
        "history_sha256": (
            "321e441f428198a3479c3a375532ae87ccd8aeabc4c0584330b6bd8cc7af98cf"
        ),
        "manifest_sha256": (
            "66c9dcf2d866a0abddc1f2a85b913fc0de20ac0c833af135ec13c4cc19e82bf6"
        ),
        "epoch": 7,
        "source_validation": {
            "photo_accuracy": 0.9700598802395209,
            "photo_macro_f1": 0.9649705082460817,
            "art_painting_accuracy": 0.9243902439024391,
            "art_painting_macro_f1": 0.92597515211018,
            "cartoon_accuracy": 0.9530916844349681,
            "cartoon_macro_f1": 0.9578542361003685,
            "mean_source_accuracy": 0.9491806028589761,
            "mean_source_macro_f1": 0.9495999654855433,
            "worst_source_accuracy": 0.9243902439024391,
            "worst_source_macro_f1": 0.92597515211018,
        },
    },
    "dan_dg_floor_0p1": {
        "method": "dan_dg_bandwidth_floor",
        "research_variant": True,
        "training_code_sha256": TRAINING_CODE_SHA256,
        "mmd_lambda": 0.1,
        "file_sha256": (
            "46e770107eb563517239d524622bc7cb121d9e166de398b692350fc80c0a453c"
        ),
        "history_sha256": (
            "50246e91eeb0d53eabb1cd8fc0901af6b5c00b23918b2aaaf8a44b333ac0617f"
        ),
        "manifest_sha256": (
            "ff7fccf71ab28d7e4fe5b22b4bfdfa5b721e2eedaaf3d9809f98900d91404242"
        ),
        "epoch": 25,
        "source_validation": {
            "photo_accuracy": 0.9640718562874252,
            "photo_macro_f1": 0.9568413478846398,
            "art_painting_accuracy": 0.9,
            "art_painting_macro_f1": 0.905926086081068,
            "cartoon_accuracy": 0.9680170575692963,
            "cartoon_macro_f1": 0.9728535317013297,
            "mean_source_accuracy": 0.9440296379522405,
            "mean_source_macro_f1": 0.9452069885556792,
            "worst_source_accuracy": 0.9,
            "worst_source_macro_f1": 0.905926086081068,
        },
    },
    "dan_dg_floor_10": {
        "method": "dan_dg_bandwidth_floor",
        "research_variant": True,
        "training_code_sha256": TRAINING_CODE_SHA256,
        "mmd_lambda": 10.0,
        "file_sha256": (
            "030adaa198612ccda2914aa283c83c12554f16a210e82ba5f4e3fcd93c4d5483"
        ),
        "history_sha256": (
            "1597de06e3976f9f80eac598f0f8c9e08b5d165d4be3f6cad26121c18866dbf6"
        ),
        "manifest_sha256": (
            "db388ebdd098b4cf89ffe4ecf44a7cb372b3aaa381e21116de221ea0d79ece31"
        ),
        "epoch": 17,
        "source_validation": {
            "photo_accuracy": 0.9550898203592815,
            "photo_macro_f1": 0.9478497033307226,
            "art_painting_accuracy": 0.875609756097561,
            "art_painting_macro_f1": 0.8746958622159983,
            "cartoon_accuracy": 0.9381663113006397,
            "cartoon_macro_f1": 0.9408272365518939,
            "mean_source_accuracy": 0.9229552959191607,
            "mean_source_macro_f1": 0.9211242673662049,
            "worst_source_accuracy": 0.875609756097561,
            "worst_source_macro_f1": 0.8746958622159983,
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object")
    return value


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assert_probe_design(saved: dict, expected_partition: dict) -> None:
    expected = {
        "status": "TASK3_SOURCE_DOMAIN_PROBE_DESIGN_LOCKED",
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "selection_rng": "numpy.default_rng_PCG64_single_stream_domain_order_P_A_C",
        "feature_representation": "raw_unnormalized_512d_pre_classifier",
    }
    for name, value in expected.items():
        if saved.get(name) != value:
            raise RuntimeError(f"Locked probe design differs: {name}")
    if saved.get("partition") != expected_partition:
        raise RuntimeError("Locked probe partition differs from deterministic reconstruction")


def assert_sharpness_design(
    saved: dict,
    expected_selected: dict[str, list[dict]],
) -> None:
    expected = {
        "status": "TASK3_SHARPNESS_BATCH_DESIGN_LOCKED",
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "selection_rng": (
            "numpy.default_rng_PCG64_independent_stream_domain_order_P_A_C"
        ),
        "seed": SEED,
        "per_domain": 32,
        "radius": SHARPNESS_RADIUS,
        "records": {
            domain: [dict(record) for record in expected_selected[domain]]
            for domain in SOURCES
        },
    }
    for name, value in expected.items():
        if saved.get(name) != value:
            raise RuntimeError(f"Locked sharpness design differs: {name}")


def assert_checkpoint_metadata(payload: dict, model_name: str) -> None:
    spec = MODEL_SPECS[model_name]
    expected = {
        "source_only_phase": True,
        "sketch_images_accessed": 0,
        "epoch": spec["epoch"],
    }
    if spec["research_variant"]:
        expected.update(
            {
                "research_variant": True,
                "primary_protocol_replacement": False,
                "final_evaluation_started": False,
            }
        )
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"{model_name} checkpoint metadata differs: {name}")
    identity = payload.get("identity", {})
    if identity.get("code_sha256") != spec["training_code_sha256"]:
        raise RuntimeError(f"{model_name} training code identity differs")
    if spec["research_variant"]:
        if identity.get("variant_id") != VARIANT_ID:
            raise RuntimeError(f"{model_name} variant identity differs")
        if identity.get("variant_protocol_version") != VARIANT_PROTOCOL_VERSION:
            raise RuntimeError(f"{model_name} variant protocol differs")
    config = identity.get("config", {})
    for name, value in {
        "run_id": model_name,
        "method": spec["method"],
        "mmd_lambda": spec["mmd_lambda"],
    }.items():
        if config.get(name) != value:
            raise RuntimeError(f"{model_name} checkpoint config differs: {name}")


def load_variant_model(
    path: Path,
    model_name: str,
    device: torch.device,
) -> tuple[PACSClassifier, dict]:
    spec = MODEL_SPECS[model_name]
    if not path.is_file() or sha256_file(path) != spec["file_sha256"]:
        raise RuntimeError(f"{model_name} checkpoint identity differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert_checkpoint_metadata(payload, model_name)
    recorded = payload.get("source_validation", {})
    for name, expected in spec["source_validation"].items():
        if name not in recorded or abs(float(recorded[name]) - expected) > 1e-12:
            raise RuntimeError(f"{model_name} recorded source metric differs: {name}")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"{model_name} checkpoint has no model state")
    model = PACSClassifier(pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, {
        "path": str(path),
        "file_sha256": spec["file_sha256"],
        "state_dict_sha256": state_dict_sha256(model.state_dict()),
        "selected_epoch": spec["epoch"],
    }


def assert_expected_metrics(model_name: str, metrics: dict) -> None:
    for name, expected in MODEL_SPECS[model_name]["source_validation"].items():
        if abs(float(metrics[name]) - expected) > 1e-12:
            raise RuntimeError(f"{model_name} re-evaluation differs: {name}")


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Variant source diagnostics already exist: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("The locked diagnostic environment requires CUDA")
    device = torch.device("cuda")
    environment = runtime_identity(device)
    if environment != EXPECTED_RUNTIME:
        raise RuntimeError(f"Diagnostic runtime differs: {environment}")
    seed_everything(SEED)

    protocol = load_source_protocol(args.protocol)
    dataset = verify_source_snapshot(args.pacs_source_root, protocol)
    if dataset.get("sketch_images_accessed") != 0:
        raise RuntimeError("Source snapshot verification reports Sketch access")

    probe_path = args.primary_diagnostics_root / "source_domain_probe_design.json"
    sharpness_path = args.primary_diagnostics_root / "sharpness_batch_design.json"
    if sha256_file(probe_path) != PROBE_DESIGN_SHA256:
        raise RuntimeError("Primary source-domain probe design identity differs")
    if sha256_file(sharpness_path) != SHARPNESS_DESIGN_SHA256:
        raise RuntimeError("Primary sharpness-batch design identity differs")

    probe_selected = select_probe_records(protocol)
    probe_partition = make_probe_partition(probe_selected)
    assert_probe_design(load_json(probe_path, "locked probe design"), probe_partition)

    sharpness_selected = select_sharpness_records(protocol)
    assert_sharpness_design(
        load_json(sharpness_path, "locked sharpness design"),
        sharpness_selected,
    )
    sharpness_images, sharpness_labels, sharpness_identifiers = load_sharpness_batch(
        sharpness_selected,
        args.pacs_source_root,
    )
    expected_sharpness_identifiers = [
        record["path"]
        for domain in SOURCES
        for record in sharpness_selected[domain]
    ]
    if sharpness_identifiers != expected_sharpness_identifiers:
        raise RuntimeError("Sharpness batch order differs from the locked design")

    results = {}
    for model_name in MODEL_ORDER:
        seed_everything(SEED)
        checkpoint = (
            args.variant_root / "training" / model_name / "best.pt"
            if MODEL_SPECS[model_name]["research_variant"]
            else args.primary_training_root / model_name / "best.pt"
        )
        model, identity = load_variant_model(checkpoint, model_name, device)
        source_validation = validate_sources(
            model,
            protocol,
            args.pacs_source_root,
            device,
        )
        assert_expected_metrics(model_name, source_validation)
        features = extract_probe_features(
            model,
            probe_selected,
            probe_partition["identifiers"],
            args.pacs_source_root,
            device,
        )
        probe = fit_domain_probe(features, probe_partition)
        before = state_dict_sha256(model.state_dict())
        sharpness = common_sharpness_proxy(
            model,
            sharpness_images.to(device, non_blocking=True),
            sharpness_labels.to(device, non_blocking=True),
        )
        after = state_dict_sha256(model.state_dict())
        if before != after:
            raise RuntimeError(f"{model_name} changed during sharpness measurement")
        results[model_name] = {
            "checkpoint": identity,
            "source_validation_reproduced": source_validation,
            "source_domain_separability": probe,
            "common_sharpness_proxy": sharpness,
        }
        del model, features
        torch.cuda.empty_cache()

    result = {
        "status": "TASK3_ADDITIONAL_SOURCE_DIAGNOSTICS_PASS",
        "scope": "missing_primary_strengths_and_bandwidth_floor_variants",
        "variant_id": VARIANT_ID,
        "variant_protocol_version": VARIANT_PROTOCOL_VERSION,
        "contains_research_variant_models": True,
        "primary_protocol_replacement": False,
        "source_only_phase": True,
        "diagnostics_completed": True,
        "final_evaluation_started": False,
        "sketch_images_accessed": 0,
        "primary_training_commit": PRIMARY_TRAINING_COMMIT,
        "primary_training_code_tree_sha256": PRIMARY_TRAINING_CODE_SHA256,
        "variant_training_commit": TRAINING_COMMIT,
        "variant_training_code_tree_sha256": TRAINING_CODE_SHA256,
        "diagnostic_code_tree_sha256": code_tree_sha256(args.code_root),
        "source_protocol_sha256": sha256_file(args.protocol),
        "source_snapshot_sha256": dataset["source_snapshot_sha256"],
        "runtime": environment,
        "reused_primary_probe_design": {
            "path": str(probe_path),
            "sha256": PROBE_DESIGN_SHA256,
            "image_count": len(probe_partition["identifiers"]),
        },
        "reused_primary_sharpness_design": {
            "path": str(sharpness_path),
            "sha256": SHARPNESS_DESIGN_SHA256,
            "image_count": len(sharpness_identifiers),
        },
        "models": results,
    }
    atomic_write_json(result, args.output)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--pacs-source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary-training-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--primary-diagnostics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
