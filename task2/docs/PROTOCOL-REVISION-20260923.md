# Task 2 protocol revision - 23 September 2026

## Status

This revision was made before any official v2 training and before any corrected-run
target-label evaluation. The previous runs remain preserved as diagnostic pilots.

## Evidence prompting the revision

| Pilot | Selected source macro-F1 | Checkpoint SHA256 | Status |
|---|---:|---|---|
| Source-only | 0.9373899735 | `034a4bb4035184e98b950d44f5afa13003a7e6d44f652f668c138cce71e8a552` | Unclipped pilot |
| DAN lambda 0.1 | 0.9483037087 | `92048571aa48cf1e29a03141520a06aa945aabdb9b492ea54822109a3f8863a5` | Unclipped pilot |
| DAN lambda 1 | 0.0506699650 | `bdad529a4c33c8ef1d84c5b999c216b66574055b327f2f869654be4ebdac9b0a` | Unclipped collapsed pilot |

The DAN lambda 1 pilot predicted class ID 6 for all 1,213 source-validation images.
Its epoch-average feature-distance median fell from an initial-batch value of about
558.76 to 6.87 in epoch 1, while its recorded epoch-average gradient norm was 45.39
and later reached 159.97. Healthy Source-only and DAN lambda 0.1 epoch-1 average norms
were approximately 11.50 and 10.91.

## TA clarification and student decision

The TA announced that exploding gradients and non-convergence may be addressed with
documented stabilization such as gradient clipping, normalization, or hyperparameter
changes. The student approved global gradient-norm clipping for every Task 2 method so
the controlled DAN study and four-method comparison retain a common training pipeline.

## Revised rule

Apply global L2 gradient-norm clipping with `max_norm=20` immediately after backward and
before each AdamW step. Apply it to the complete joint trainable parameter set for all
six configurations. Record pre-clipping norm, post-clipping norm, and clipped-step
fraction. All other D1-D15 decisions remain unchanged.

The official v2 runs use a new output root. Pilot checkpoints and results are excluded
from the official comparison and remain available only as an audit trail.
