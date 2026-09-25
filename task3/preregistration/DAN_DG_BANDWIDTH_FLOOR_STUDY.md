# Preregistered Supplementary Study: Initialization-Anchored MMD Bandwidth Floor

Status: approved before implementation and before any Task 3 access to Sketch  
Approval date: 25 September 2026  
Variant ID: `dan_dg_initial_bandwidth_floor_v1`  
Protocol version: `task3-research-bandwidth-floor-2026-09-25-v1`

## Scientific position

This is a source-only, supplementary research variant. It does not replace, revise, or
invalidate the assignment-prescribed primary DAN-DG runs. In particular, the original
lambda-1 result remains the main DAN-DG result required by the assignment. The new
variant is explicitly post-diagnostic and must be reported as such if discussed.

## Source-only evidence motivating the intervention

The completed prescribed lambda-1 run showed all of the following without using Sketch:

- the three epoch-average median squared feature distances fell from approximately
  `0.0265-0.0302` in epoch 1 to approximately `0.0000406-0.0000414` in epoch 8;
- 224 of 235 epoch-1 updates and every update in epochs 2-8 required global clipping;
- the average pre-clipping total-gradient norm rose from `156.746653` to `782.310502`;
- the selected mean source-validation macro-F1 was `0.8693286334613551`, below the
  frozen ERM value `0.9426262342459099`; and
- the common source-only sharpness delta was `94.29026794433594`, compared with
  `0.24889972805976868` for ERM.

The controlled lambda-0.1 run, with every other training setting unchanged, selected a
mean source-validation macro-F1 of `0.9462297763360951` and clipped only 70 of 2350
updates. These observations motivate the hypothesis that adaptive bandwidth collapse,
rather than an implementation mismatch, destabilized the stronger alignment runs.

## Single controlled change

The locked current-batch median `m_current` is still calculated exactly as in the
primary implementation. For each unordered source pair, the supplementary variant uses:

```text
m_effective = max(m_current, m_initial_floor[pair])
bandwidths = {0.5, 1, 2} * m_effective
```

The kernel convention, three-kernel sum, V-statistic estimator, feature normalization,
off-diagonal-zero policy, and mean over the three source pairs are otherwise unchanged.

## Floor calibration

Before any variant training update, load the exact common initialization. Use the
source training records only, ignore their labels, and process 235 deterministic
calibration updates with eight records per domain. Use the locked validation transform
(resize, center crop, ImageNet normalization) so calibration has no random augmentation.
The domain samplers use the exact deterministic epoch-zero source sampler seeds.

For each source pair and each calibration update, calculate the original current-batch
median squared distance. The pair floor is the median of the 235 recorded medians. The
odd sample count makes the reduction unambiguous. The multiplier is exactly 1.0, so no
floor-strength hyperparameter is introduced. Freeze the resulting three values and
reuse them unchanged for every lambda.

## Controlled study

Run `lambda_DG` in `{0.1, 1, 10}`. Every condition starts from the same authenticated
common initialization. All primary settings remain fixed: data and split, source-only
embargo, model, preprocessing, random augmentation during training, source sampler,
eight examples per domain, 235 updates per epoch, BatchNorm policy, AdamW settings,
global gradient clipping, epoch budget, patience, and source-validation selection.

Run order is lambda 1, then 0.1, then 10. The order is administrative only: all three
settings and the shared floors are frozen before the first variant run. Preserve every
finite result; do not select a lambda using Sketch.

## Required evidence

For each epoch save classification loss, pairwise MMD, current and effective pair
medians, floor activation fractions, gradient norms, clipping fraction, and all source
validation metrics. After all runs, repeat the already approved source-domain probe and
common sharpness proxy on the three selected variant checkpoints. No Sketch image or
label may be opened during calibration, training, selection, or diagnostics.

## Interpretation boundary

Because the floor changes the assignment-prescribed adaptive bandwidth rule, this
variant is not the primary PDF-compliant DAN-DG implementation. It is evidence about a
specific stabilization hypothesis. Any improvement supports the claim that bandwidth
contraction contributed to instability; it does not prove that this is the only cause.
