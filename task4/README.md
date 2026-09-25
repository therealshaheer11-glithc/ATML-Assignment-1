# Task 4 - Open-Set Recognition

This directory implements the PA-required CIFAR-10 open-set study with Vanilla,
GCSC, and PROSER. The exact locked settings and implementation choices are recorded
in `docs/PROTOCOL_AND_DECISIONS.md`.


## Completed experiment

The seed-6304 experiment is complete. The selected checkpoints,
training histories, frozen evaluation lock, final tables, figure,
and failure analysis are published under `results/` and
`provenance/`.

The final evaluation used 10,000 CIFAR-10 test images, 800 fixed
near CIFAR-100 test unknowns, and 800 fixed far CIFAR-100 test
unknowns. No post-evaluation model selection or tuning was
performed. The optional RPL extension was not included.

## Safety boundary

Training, checkpoint selection, score design, Mahalanobis fitting, and threshold
selection use CIFAR-10 only. `freeze_evaluation.py` does not import the CIFAR-100
module. The fixed CIFAR-100 test unknowns become accessible only inside
`evaluate_osr.py`, after it authenticates the frozen evaluation lock and all selected
checkpoints. CIFAR-100 training data is never constructed.

## Implemented methods

| Method | Training |
| --- | --- |
| Vanilla | Ordinary ten-class cross-entropy |
| GCSC | Vanilla recipe plus the PA-locked RandAugment operation |
| PROSER | Five classifier placeholders and layer2 manifold-mixup data placeholders |

The optional reciprocal-point extension is intentionally not included.

## Local verification

From the repository root:

```bash
python -m pytest -q tests task3/tests task4/tests
python -m task4.preflight
```

## Colab execution order

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

After reviewing all three CIFAR-10-only runs, freeze scores and thresholds. Do not
run the final evaluation command before reviewing `evaluation_lock.json`.

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

Only after the lock is approved, run the one-time fixed evaluation:

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

The final command saves machine-readable tables, cached logits/features, six fixed
failure examples, and the required compact score-distribution figure. Do not use the
unknown results to retrain, reselect, or redefine anything.

