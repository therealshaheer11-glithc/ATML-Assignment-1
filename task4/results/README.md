# Task 4 Final Results

## Experiment status

Task 4 training, target-free evaluation locking, and the
one-time fixed CIFAR-100 test evaluation are complete.

- Seed: 6304
- Known test set: 10,000 CIFAR-10 test images
- Near unknowns: 800 fixed CIFAR-100 test images
- Far unknowns: 800 fixed CIFAR-100 test images
- RPL included: no
- Evaluation-lock SHA-256:
  `d71fee7c64e8efc1f9622bffd08b7bae85fa2227c9a49c1ba4d7af759669f6ac`

No model, checkpoint, score, threshold, or calibration value
was changed after unknown-data access.

## Selected checkpoints

| Method | Best epoch | Validation accuracy | Test accuracy |
| --- | ---: | ---: | ---: |
| Vanilla | 97 | 0.9510 | 0.9454 |
| GCSC | 99 | 0.9552 | 0.9494 |
| PROSER | 1 | 0.9496 | 0.9414 |

## Vanilla score comparison

| Score | Near AUROC | Far AUROC | Overall AUROC |
| --- | ---: | ---: | ---: |
| MSP | 0.8103 | 0.8962 | 0.8533 |
| MLS | 0.7916 | 0.8983 | 0.8449 |
| Energy | 0.7919 | 0.8994 | 0.8457 |
| Mahalanobis | 0.7965 | 0.9139 | 0.8552 |

## Trained-model comparison

| Method and score | Near AUROC | Far AUROC | Overall AUROC |
| --- | ---: | ---: | ---: |
| Vanilla + MLS | 0.7916 | 0.8983 | 0.8449 |
| GCSC + MLS | 0.7989 | 0.9085 | 0.8537 |
| PROSER + MLS | 0.7984 | 0.8943 | 0.8463 |
| PROSER placeholder | 0.7915 | 0.8973 | 0.8444 |

## Included evidence

- Complete machine-readable final results
- Both required CSV comparison tables
- Fixed-threshold failure analysis
- Score-distribution figure
- Training histories and completion summaries
- Post-evaluation per-class analysis ([`extra_analysis.json`](extra_analysis.json), produced by [`../analysis/task4_extra_analysis.py`](../analysis/task4_extra_analysis.py) from the saved outputs)
- Fixed CIFAR-10 split manifest
- Evaluation lock and Mahalanobis statistics
- Runtime and dataset provenance

Large `.pt` checkpoints and cached logits/features remain in
Google Drive and are intentionally excluded from Git.
