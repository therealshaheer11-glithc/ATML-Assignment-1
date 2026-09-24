# Task 3 - Domain Generalization

This directory implements the approved source-only training stage for Task 3 of
ATML Programming Assignment 1. The authoritative choices and their rationale are in
`docs/TASK3_PROTOCOL_AND_DECISIONS.md`; this README is only the execution guide.

## Safety boundary

During training, model selection, unit testing, and source-side diagnostics, this code
may access only Photo, Art Painting, and Cartoon. It must not traverse, load, inspect,
or extract Sketch. `data.py` accepts only the mechanically derived source-only
protocol and verifies its SHA-256 identity before constructing any dataset.

Do not add Sketch paths or a target loader to `train.py`. Final Sketch evaluation will
be implemented as a separate component only after every configuration and checkpoint
has been frozen in the final experiment-lock manifest and the student has explicitly
authorized target access.

## Implemented runs

| Run ID | Method | Role |
|---|---|---|
| `dan_dg_0p1` | DAN-DG, lambda 0.1 | controlled study |
| `dan_dg_1` | DAN-DG, lambda 1 | main comparison and controlled study |
| `dan_dg_10` | DAN-DG, lambda 10 | controlled study |
| `sam` | standard non-adaptive SAM, rho 0.05 | main comparison |

The Task 2 ERM checkpoint is reused without retraining.

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

## Verification before training

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

## Training commands

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

## Source-only diagnostics after training

After all four training configurations have completed and their checkpoints have been
reviewed, run the deterministic source diagnostics before creating the final experiment
lock. The diagnostic code never accepts a Sketch path.

```bash
python -m task3.evaluation.run_source_diagnostics \
  --code-root /content/atml_pa1_task3_source \
  --pacs-source-root /content/task3_pacs_sources_v1 \
  --protocol /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/source_protocol/pacs_sources_seed6304.json \
  --erm-checkpoint /content/drive/MyDrive/ATML-PA1/task2_corrected_normalized_v3_20260923/source_only/best.pt \
  --training-root /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/training \
  --output /content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/source_diagnostics/source_diagnostics.json
```

The command saves the exact 1,002-image source-domain probe design, its shared
domain-stratified 70/30 partition, the fixed 96-image sharpness batch, reproduced
source-validation metrics, source-domain separability, and the common radius-0.05
sharpness proxy for ERM, prescribed DAN-DG lambda 1, and SAM. It uses raw 512-D
features for the probe, fits `StandardScaler` only on the probe-training partition,
and never loads Sketch.

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
