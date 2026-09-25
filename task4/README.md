# Task 4 - Open-Set Recognition

This directory implements the PA-required CIFAR-10 open-set study with Vanilla,
GCSC, and PROSER. The exact locked settings and implementation choices are recorded
in the [Task 4 protocol and decisions](docs/PROTOCOL_AND_DECISIONS.md).

## Completed experiment

The seed-6304 experiment is complete. The selected-checkpoint records,
training histories, frozen evaluation lock, final tables, figure,
and failure analysis are published under [`results/`](results/) and
[`provenance/`](provenance/).

The large selected `.pt` checkpoint files remain in Google Drive and are intentionally
excluded from Git. Their SHA-256 identities are preserved in the published evaluation
lock.

The final evaluation used 10,000 CIFAR-10 test images, 800 fixed near CIFAR-100 test
unknowns, and 800 fixed far CIFAR-100 test unknowns. No post-evaluation model
selection or tuning was performed. The optional RPL extension was not included.

The final metrics and published evidence are indexed in the
[Task 4 results guide](results/README.md).

## Safety boundary

Training, checkpoint selection, score design, Mahalanobis fitting, and threshold
selection use CIFAR-10 only. `freeze_evaluation.py` does not import the CIFAR-100
module. The fixed CIFAR-100 test unknowns become accessible only inside
`evaluate_osr.py`, after it authenticates the frozen evaluation lock and all selected
checkpoints. CIFAR-100 training data is never constructed.

The completed one-time CIFAR-100 evaluation is a historical action. It should not be
repeated merely to verify the repository. Use the published lock, completion record,
tables, figure, and failure analysis for verification.

## Implemented methods

| Method | Training |
|---|---|
| Vanilla | Ordinary ten-class cross-entropy |
| GCSC | Vanilla recipe plus the PA-locked RandAugment operation |
| PROSER | Five classifier placeholders and layer2 manifold-mixup data placeholders |

The optional reciprocal-point extension is intentionally not included.

## Local verification

From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests task3/tests task4/tests
python -m task4.preflight
```

The lightweight development requirements install pytest. Running the Task 4 code also
requires the scientific and machine-learning dependencies listed in
[`requirements.txt`](requirements.txt). The exact runtime used for the completed
experiment is recorded in
[`provenance/runtime_preflight.json`](provenance/runtime_preflight.json).

## Historical Colab execution order

The commands below document how the completed experiment was executed. They are
retained for reproducibility and inspection; the final one-time unknown evaluation
should not be rerun as part of ordinary repository verification.

Use a persistent output root outside Git, for example:

```text
/content/drive/MyDrive/ATML-PA1/task4_open_set_20260925
```

First create the fixed CIFAR-10 split:

```bash
python -m task4.data.make_splits \
  --data-root /content/data \
  --output /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/splits/cifar10_seed6304.json \
  --download
```

Train Vanilla and GCSC independently:

```bash
python -m task4.train \
  --config task4/configs/vanilla.yaml \
  --data-root /content/data \
  --split-manifest /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/splits/cifar10_seed6304.json \
  --output /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/vanilla

python -m task4.train \
  --config task4/configs/gcsc.yaml \
  --data-root /content/data \
  --split-manifest /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/splits/cifar10_seed6304.json \
  --output /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/gcsc
```

Then initialize PROSER from the selected Vanilla checkpoint:

```bash
python -m task4.train \
  --config task4/configs/proser.yaml \
  --data-root /content/data \
  --split-manifest /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/splits/cifar10_seed6304.json \
  --vanilla-checkpoint /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/vanilla/best.pt \
  --output /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/proser
```

After reviewing all three CIFAR-10-only runs, freeze the score definitions and
thresholds. The final evaluation must not begin until `evaluation_lock.json` has been
reviewed and approved.

```bash
python -m task4.freeze_evaluation \
  --data-root /content/data \
  --split-manifest /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/splits/cifar10_seed6304.json \
  --vanilla-config task4/configs/vanilla.yaml \
  --vanilla-checkpoint /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/vanilla/best.pt \
  --gcsc-config task4/configs/gcsc.yaml \
  --gcsc-checkpoint /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/gcsc/best.pt \
  --proser-config task4/configs/proser.yaml \
  --proser-checkpoint /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/proser/best.pt \
  --output /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/evaluation_lock
```

For historical reference, the authorized one-time evaluation command was:

```bash
python -m task4.evaluate_osr \
  --data-root /content/data \
  --split-manifest /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/splits/cifar10_seed6304.json \
  --evaluation-lock /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/evaluation_lock/evaluation_lock.json \
  --vanilla-checkpoint /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/vanilla/best.pt \
  --gcsc-checkpoint /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/gcsc/best.pt \
  --proser-checkpoint /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/training/proser/best.pt \
  --output /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/final_evaluation \
  --download
```

That command saved machine-readable tables, cached logits and features, six fixed
failure examples, and the compact score-distribution figure. The published repository
contains the final tables, histories, figure, failure analysis, split, lock, and
provenance records. Large checkpoints and cached logits and features remain outside
Git.

No unknown result may be used to retrain a model, reselect a checkpoint, change a
threshold, redefine a score, or otherwise alter the locked experiment.
