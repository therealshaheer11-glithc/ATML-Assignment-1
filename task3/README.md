# Task 3 - Domain Generalization

This directory implements the approved source-only training stage for Task 3 of
ATML Programming Assignment 1. The authoritative choices and their rationale are in
[`docs/TASK3_PROTOCOL_AND_DECISIONS.md`](docs/TASK3_PROTOCOL_AND_DECISIONS.md).
This README provides the experiment overview and historical execution commands.

For verification of the published evidence and reproduction of historical stages, use
the [pinned reproduction guide](docs/REPRODUCTION.md). Published final results and
training histories are indexed in the [Task 3 results guide](results/README.md).

## Safety boundary

During training, model selection, unit testing, and source-side diagnostics, this code
may access only Photo, Art Painting, and Cartoon. It must not traverse, load, inspect,
or extract Sketch. `data.py` accepts only the mechanically derived source-only
protocol and verifies its SHA-256 identity before constructing any dataset.

Do not add Sketch paths or a target loader to `train.py`. The completed final Sketch
evaluation is isolated in `evaluation/final_sketch.py`; it ran only after every
configuration and checkpoint had been frozen in the final experiment-lock manifest
and the student had explicitly authorized target access.

## Implemented runs

| Run ID | Method | Role |
|---|---|---|
| `dan_dg_0p1` | DAN-DG, lambda 0.1 | controlled study |
| `dan_dg_1` | DAN-DG, lambda 1 | main comparison and controlled study |
| `dan_dg_10` | DAN-DG, lambda 10 | controlled study |
| `sam` | standard non-adaptive SAM, rho 0.05 | main comparison |

The Task 2 ERM checkpoint is reused without retraining.

## Final evaluation status

Blocks 15 and 16 completed successfully at repository commit
`31618caebaf42acb18dd657407d22f03b4a2464f`. Block 15 froze all eight selected
checkpoints without accessing Sketch. Block 16 then evaluated those checkpoints once
on all 3,929 locked Sketch images. No target label was used for training or model
selection, and no checkpoint was changed after the evaluation.

The exact completion records, aggregate results, and artifact hashes are documented in
the [Task 3 execution log](provenance/RUN_LOG.md). The executed notebook is preserved
unchanged at
[`provenance/notebooks/ATML_PA_TASK3.ipynb`](provenance/notebooks/ATML_PA_TASK3.ipynb).

The published final results, per-example predictions, selected failure examples, and
training histories are indexed in the [Task 3 results guide](results/README.md).

## Supplementary bandwidth-floor research study

After every prescribed run and the locked source-only diagnostics were complete, a
separate research study was approved to test whether initialization-anchored bandwidth
floors stabilize DAN-DG. This study is not a replacement for the PDF-prescribed
adaptive-bandwidth runs. Its exact motivation, single controlled change, invariants,
calibration, comparison plan, and interpretation boundary are recorded in
[`docs/DAN_DG_BANDWIDTH_FLOOR_RESEARCH_VARIANT.md`](docs/DAN_DG_BANDWIDTH_FLOOR_RESEARCH_VARIANT.md)
and
[`preregistration/DAN_DG_BANDWIDTH_FLOOR_STUDY.md`](preregistration/DAN_DG_BANDWIDTH_FLOOR_STUDY.md).

The study uses run IDs `dan_dg_floor_0p1`, `dan_dg_floor_1`, and
`dan_dg_floor_10`. All three reuse one source-only calibration file and are written to
a distinct research-variant output directory. They must never overwrite or be
presented as the original `dan_dg_0p1`, `dan_dg_1`, or `dan_dg_10` runs.

## Important implementation identities

- Task 2 source protocol SHA-256:
  `e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74`
- Task 3 source-only protocol SHA-256:
  `626d8517b44ad50c0219adf49e827de6538561386791bed29a9153a589cd6abc`
- Task 3 source snapshot SHA-256:
  `8ded350769ee15739f8420e755e50ff4377068a4f54ab1c0ba39d5b125e658d2`
- Common initialization state SHA-256:
  `4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3`
- Task 2 ERM checkpoint file SHA-256:
  `3d28a223e4b97b323cb3a20dcb5b7577af96631f2e6ef1f2bc99d53d85761327`
- Locked shared MMD implementation SHA-256:
  `cfe0b1d9c22d7f492ea5e8f76732fbabf21c86cb53f24759af65fda09f9bfbcc`
- DAN-DG study preregistration SHA-256:
  `e8377b762cc1872b6e814e565869414320cff94bbacd69fe097d70b29d4146fb`

## Historical verification before prescribed training

The commands in this section describe the original prescribed training stage. First
check out commit:

```text
19208b4c62acb980fb3246f30e062784b90d8dfc
```

The complete snapshot sequence is documented in the
[pinned reproduction guide](docs/REPRODUCTION.md). The preserved notebook blocks use
snapshot-specific test counts and code hashes and must not be combined with the latest
`main` checkout.

Run the target-free unit tests from the repository root:

```bash
python -m unittest discover -s task3/tests -v
```

Then run the code preflight in the exact locked Colab runtime:

```bash
python -m task3.preflight \
  --code-root /content/atml_pa1_task3_source \
  --pacs-source-root /content/task3_pacs_sources_v1 \
  --protocol /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/source_protocol/pacs_sources_seed6304.json \
  --initialization /content/drive/MyDrive/ATML-PA1/task2_corrected_normalized_v3_20260923/initialization/resnet18_v1_seed6304_common.pt \
  --erm-checkpoint /content/drive/MyDrive/ATML-PA1/task2_corrected_normalized_v3_20260923/source_only/best.pt \
  --preregistration /content/atml_pa1_task3_source/task3/preregistration/DAN_DG_STRENGTH_EXPECTATION.md \
  --output /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/provenance/code_preflight.json
```

Training must not begin unless both checks pass.

## Historical prescribed training commands

Run exactly one configuration at a time. The output root is shared so that every run
must agree on the source protocol, source snapshot, code, initialization,
preregistration, and runtime identities.

```bash
python -m task3.train \
  --run-id dan_dg_1 \
  --pacs-source-root /content/task3_pacs_sources_v1 \
  --protocol /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/source_protocol/pacs_sources_seed6304.json \
  --initialization /content/drive/MyDrive/ATML-PA1/task2_corrected_normalized_v3_20260923/initialization/resnet18_v1_seed6304_common.pt \
  --preregistration /content/atml_pa1_task3_source/task3/preregistration/DAN_DG_STRENGTH_EXPECTATION.md \
  --code-preflight /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/provenance/code_preflight.json \
  --output /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/training
```

Substitute `dan_dg_0p1`, `dan_dg_10`, or `sam` only when intentionally running that
approved configuration. Add `--resume` only after confirming that the matching run's
`last.pt` is a valid completed-epoch checkpoint. A partial epoch is never resumed.

Each run writes:

- `history.csv`, containing losses, gradient diagnostics, and source-validation
  metrics for every completed epoch;
- `best.pt`, selected only by strict improvement in mean source-validation macro-F1;
- `best_source_validation.json`;
- `last.pt`, containing the complete resumable state from the last completed epoch;
  and
- `run.json`, written after early stopping or the 30-epoch budget.

The original published histories are available under
[`results/training/prescribed/`](results/training/prescribed/).

## Historical source-only diagnostics

After the prescribed training configurations completed and their checkpoints were
reviewed, deterministic source-only diagnostics were run before the final experiment
lock. The diagnostic code never accepted a Sketch path.

The exact diagnostic snapshots, including the later complete eight-model diagnostic
stage, are identified in the
[pinned reproduction guide](docs/REPRODUCTION.md). The complete diagnostic stage used
commit:

```text
136e28b556d38c99e486e2d53ddff33d921fbfc6
```

The original diagnostic command was:

```bash
python -m task3.evaluation.run_source_diagnostics \
  --code-root /content/atml_pa1_task3_source \
  --pacs-source-root /content/task3_pacs_sources_v1 \
  --protocol /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/source_protocol/pacs_sources_seed6304.json \
  --erm-checkpoint /content/drive/MyDrive/ATML-PA1/task2_corrected_normalized_v3_20260923/source_only/best.pt \
  --training-root /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/training \
  --output /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/source_diagnostics/source_diagnostics.json
```

The diagnostics saved the exact 1,002-image source-domain probe design, its shared
domain-stratified 70/30 partition, the fixed 96-image sharpness batch, reproduced
source-validation metrics, source-domain separability, and the common radius-0.05
sharpness proxy. The probe used raw 512-D features and fitted `StandardScaler` only on
the probe-training partition. No diagnostic stage loaded Sketch.

## Completed one-time target evaluation

The final experiment lock and one-time Sketch evaluation were completed at repository
commit:

```text
31618caebaf42acb18dd657407d22f03b4a2464f
```

This evaluation is a completed historical action and should not be rerun merely to
verify the repository. Use the published lock, completion records, predictions, and
machine-readable results for verification. No target result may be used to retrain,
reselect, replace, or reorder a checkpoint.

## Failure policy

Any non-finite value, zero SAM gradient, invalid MMD bandwidth, changed artifact hash,
changed runtime, changed BatchNorm buffer, changed run identity, or other failed check
stops the run. Preserve the artifacts and obtain explicit approval before changing any
setting. Never use Sketch evidence to diagnose or repair source training.

## Dependencies and attribution

The implementation uses PyTorch and torchvision's `resnet18` and
`ResNet18_Weights.IMAGENET1K_V1`, Pillow for image decoding, NumPy for deterministic
sampling, and scikit-learn for metrics. The MMD implementation in `shared/mmd.py` is
the repository's locked Task 2 implementation and is reused unchanged as required by
the assignment.
