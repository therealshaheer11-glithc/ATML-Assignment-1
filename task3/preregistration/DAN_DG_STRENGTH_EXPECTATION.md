# Task 3 DAN-DG Strength Study: Preregistered Expectation

Status: APPROVED BEFORE TASK 3 TRAINING AND BEFORE TASK 3 SKETCH ACCESS  
Date: 24 September 2026  
Seed: 6304

The controlled study varies `lambda_DG` over `{0.1, 1, 10}`. Every other data,
initialization, optimization, checkpoint-selection, diagnostic, and evaluation setting
is fixed. The `lambda_DG = 1` run remains the prescribed main-comparison result and may
not be replaced by a post-hoc winner.

## Expected source performance

Increasing `lambda_DG` strengthens the pressure to make Photo, Art Painting, and
Cartoon marginal feature distributions similar. Weak alignment should behave more like
ERM, while excessive alignment may remove class-discriminative information and reduce
mean-source or worst-source accuracy and macro-F1. No monotonic source-performance
improvement is assumed.

## Expected source-domain separability

Stronger pairwise MMD pressure should generally make the observed source domains harder
for the fixed multinomial logistic-regression probe to distinguish. Therefore source-
domain separability is expected to decrease as `lambda_DG` increases. A lower score is
not, by itself, evidence that class information has been preserved.

## Expected MMD behavior

The optimized pairwise MMD penalty is expected to be lower under stronger alignment
pressure, although optimization dynamics and the simultaneous classification objective
may prevent a perfectly monotonic selected-checkpoint ordering. MMD must be interpreted
alongside source classification, prediction distributions, gradient diagnostics, and
source-domain separability.

## Expected Sketch performance

Sketch performance may be non-monotonic. Moderate observed-source alignment could
transfer if it suppresses nuisance domain variation, while excessive marginal
alignment could harm transfer by mixing class structure. No specific `lambda_DG` value
is preregistered as the expected Sketch winner, and no Sketch outcome may alter the
main setting, checkpoints, or design.

## Information restriction

This study was selected to examine the alignment mechanism and to support a controlled
comparison with target-aware Task 2 DAN. It was not selected using Task 2 Sketch results.
Task 3 Sketch images and labels remain unavailable until the final experiment lock.

