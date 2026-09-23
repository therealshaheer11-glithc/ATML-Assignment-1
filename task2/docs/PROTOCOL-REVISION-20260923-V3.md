# Task 2 v3 protocol revision - 23 September 2026

## Status

This revision was approved after the clipped v2 DAN lambda 1 stability gate and before
any v3 training or corrected-run target-label evaluation. The v1 and v2 outputs remain
preserved as diagnostic pilots and are excluded from the official comparison.

## Source-only evidence prompting the revision

The clipped v2 DAN lambda 1 run completed six epochs and selected epoch 1 using source
validation only. Its best mean source-validation macro-F1 was 0.1161215680 and its
selected-checkpoint SHA256 was
`572f2c40d9498e779e2e56ab4de4fb882b13d264eea97e838a42939b829f9761`.

The epoch-average median squared distance used by MMD fell from 6.8731 at epoch 1 to
0.0002 at epoch 2 and rounded to 0.0000 thereafter. After epoch 1, 93-98% of updates
required clipping. The selected checkpoint predicted class ID 5 for 1,078 and class ID
6 for 135 of the 1,213 pooled source-validation images, predicting no other class. This
establishes feature-scale collapse and failed source classification without consulting
Sketch labels.

## Approved v3 rule

For DAN lambda 0.1, 1, and 10, L2-normalize each source and target feature vector before
passing it to the unchanged three-kernel MMD function. Classification continues to use
the original unnormalized 512-dimensional feature. Do not add an epsilon or silently
clamp the norm; stop on a zero or non-finite norm.

All earlier MMD choices remain fixed: the literal empirical mean-embedding expression,
within-domain diagonals, strict-upper-triangle bandwidth candidates, retained
off-diagonal zeros, detached current-batch median, bandwidth factors 0.5/1/2, and
`exp(-squared_distance/(2*bandwidth))`. Global L2 gradient clipping at 20 also remains
active for every method. No learning rate, optimizer, initialization, split, sampling,
augmentation, stopping, or checkpoint-selection setting changes.

## Authorization and scope

The course TA explicitly permitted documented normalization and gradient clipping to
stabilize non-convergent training. The student approved this MMD-only normalization after
reviewing the source diagnostic. No target labels were accessed. Task 3 DAN-DG must use
the same normalized MMD inputs so Task 2 and Task 3 continue to share one discrepancy
implementation.
