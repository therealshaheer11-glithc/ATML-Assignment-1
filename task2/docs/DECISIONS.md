# Task 2 corrected rerun: locked decision record

Status: revised and re-locked before official training. The earlier Task 2 attempt is
preserved on the GitHub branch `archive/task2-attempt-1-20260923`. The first unclipped
corrected runs of Source-only, DAN 0.1, and DAN 1 are retained as diagnostic pilots and
excluded from the official comparison. Earlier target results are known, but they must
not be used to revise this protocol or select corrected checkpoints.

The assignment-mandated settings are summarized separately in `ASSIGNMENT-PROTOCOL.md`.
This file records the implementation choices that the assignment leaves open. D1 and
D2-D15 were approved by the student in conversation. D4 follows a TA response that
either listed RBF convention is acceptable.

## D1 - Controlled study

Use the DAN strength study with `lambda_MMD` in `{0.1, 1, 10}`. Reuse the `lambda=1`
run in the main four-method comparison. All other settings remain fixed.

## D2 - Finite-batch MMD expression

The assignment specifies:

```text
L_DAN = L_cls + lambda_MMD
        || E_s[phi(F(x_s))] - E_t[phi(F(x_t))] ||_H^2
```

Implement its displayed squared distance between empirical kernel mean embeddings
literally on each batch:

```text
MMD^2 = mean(K_ss) + mean(K_tt) - 2 mean(K_st)
```

Equivalently, for source features `s_1...s_ns` and target features `t_1...t_nt`:

```text
(1/ns^2) sum_i sum_j k(s_i,s_j)
+ (1/nt^2) sum_i sum_j k(t_i,t_j)
- (2/(ns nt)) sum_i sum_j k(s_i,t_j)
```

The within-source and within-target kernel diagonals are included. This is the literal
finite-batch plug-in form of the squared empirical mean-embedding distance shown in the
assignment. It is sometimes called the biased or V-statistic estimator; the assignment
itself does not use that label.

## D3 - Distances used for the bandwidth median

For the combined 24-source/24-target feature batch, use each distinct pair once via the
strict upper triangle (`i < j`). Exclude diagonal self-distances. Retain every
off-diagonal zero distance, following the TA clarification. Do not filter pairs merely
because their distance is zero.

## D4 - RBF convention

For bandwidth `b`, use:

```text
k_b(x,y) = exp(-||x-y||^2 / (2b))
```

The three bandwidths are `b in {0.5m, 1m, 2m}`, where `m` is the D3 median squared
distance. The TA confirmed that this convention and the alternative without the factor
of two are both acceptable. This convention was selected because it is the standard
Gaussian RBF form when `b` is interpreted as the variance and gives the median squared
distance a direct scale interpretation. The batch median is detached from autograd so
the network cannot reduce the objective by differentiating through its bandwidth.

## D5 - Zero or invalid bandwidth

If the retained median is non-positive or non-finite, stop the run with a diagnostic.
Do not discard more zero pairs, clamp silently, or substitute an arbitrary bandwidth.

## D6 - Source epoch and cycling

One source epoch contains
`max(ceil(number_of_training_examples_in_domain / 8))` updates across the three source
domains. The smaller source-domain loaders and the 24-example target loader cycle as
needed. Each cycle begins a new deterministic shuffled permutation. Source-only uses the
same source epoch definition and eight examples from each source domain per update.
Use the fixed concatenation order Photo, Art Painting, Cartoon. Within each domain,
sample without class reweighting: start from a seeded random permutation and begin a
fresh seeded permutation whenever that loader cycles. Target sampling follows the same
cycling rule. No sample is dropped to make an epoch divide evenly.

## D7 - AdamW completion

Use AdamW with the assignment's learning rate `1e-4` and weight decay `1e-4`, plus
`betas=(0.9,0.999)`, `eps=1e-8`, `amsgrad=False`, and a constant learning rate. Use one
parameter group for every trainable parameter, including biases and BatchNorm affine
parameters. Adaptation discriminators join the same optimizer group. There is one joint
forward/backward/optimizer step per update and no separate discriminator-only step.
Both class and domain cross-entropy use the ordinary mean reduction, with no class
weights and no label smoothing. Domain labels are source `0` and target `1`. AdamW's
loop implementation is fixed to the non-foreach, non-fused path so a runtime upgrade
cannot silently switch optimizer kernels.

## D8 - Numerical stability policy (revised before official runs)

Use float32 without autocast, GradScaler, accumulation, a scheduler, or warmup. Apply
global L2 gradient-norm clipping with `max_norm=20` to every Task 2 configuration before
each AdamW update. The clipped parameter set is the complete joint trainable set: model
parameters and, for DANN/CDAN, discriminator parameters. Use the deterministic
non-foreach clipping path. Log both the pre-clipping and post-clipping norms and the
fraction of steps clipped. Stop immediately on a non-finite loss or gradient.

Reason for revision: after the original D8 policy was locked, a source-only diagnostic
showed DAN lambda 1 collapsing to one class, with epoch-average gradient norms from 45
to 160 while healthy Source-only and DAN lambda 0.1 were near 11. The course TA then
explicitly permitted documented gradient clipping, normalization, or hyperparameter
changes to stabilize non-convergent runs. The student approved global clipping at 20
for all six configurations. The threshold is above the observed healthy epoch averages
and below the failed run's abnormal range. No corrected-run target labels were accessed
when making this revision.

## D9 - Common initialization

Create one ResNet-18 ImageNet V1 plus seven-class-head initial state under seed 6304,
hash it, and load that exact state for every method. Initialize DANN/CDAN discriminators
deterministically under seed 6304. Method comparisons must not differ because of class
head initialization.

## D10 - Gradient-reversal progress

For DANN and CDAN, compute progress against the complete planned 30-epoch update budget:

```text
p = global_update_index / (30 * updates_per_epoch - 1)
alpha(p) = 2 / (1 + exp(-10p)) - 1
```

Early stopping ends the fixed schedule at its current value; it does not rescale the
schedule using the eventual stopping epoch.

## D11 - Checkpoint ties

Evaluate source validation after each complete epoch. A checkpoint replaces the current
best only when unweighted mean source-domain macro-F1 is strictly greater. Exact ties
retain the earlier checkpoint and count toward patience five. There is no epoch-zero
selection candidate.
Validation is deterministic, unshuffled, and uses batches of 64. Because BatchNorm is
in evaluation mode, validation batch composition does not alter its running statistics.

## D12 - Domain-separability probe

After all checkpoints are frozen, use all pooled source-validation features and sample
the same number of target features with seed 6304. Create the required stratified 70/30
binary-domain split. Fit `StandardScaler` on the 70% probe-training features only, then
fit balanced `LogisticRegression(C=1, solver="lbfgs", max_iter=2000,
random_state=6304)`. Report ordinary held-out accuracy; chance is 50%.

## D13 - Macro-F1 convention

For every macro-F1 calculation, pass all seven class IDs explicitly and assign zero to
an undefined class F1 (`zero_division=0`). A method that never predicts a class must not
obtain an inflated score by having that class omitted.

## D14 - Augmentation details

Decode with Pillow and convert to RGB. Resize directly to 256 by 256 using bilinear
interpolation with antialiasing. Training uses `RandomCrop(224)` without padding followed
by horizontal flipping with probability 0.5. Apply this same training transform to
source and unlabeled target images. Validation/evaluation uses `CenterCrop(224)` with no
flip. Apply ImageNet V1 normalization after tensor conversion.

## D15 - Interruption recovery

Write resumable state only after a complete epoch. Save model, discriminator, optimizer,
selection state, completed epoch, histories, RNG states, configuration, split identity,
initialization identity, code identity, environment, and device. If execution stops
mid-epoch, discard that incomplete epoch and replay it from the last completed epoch
using its deterministic epoch-specific data seeds. Resume only with matching code,
configuration, split, GPU type, and core software versions; otherwise start that run in
a new directory.

Use two data-loader workers, pinned memory on CUDA, explicit worker generators, and
epoch-specific sampler/worker seeds. Disable TF32 and cuDNN benchmarking; enable
deterministic algorithms. Reset the global seed after method-specific module creation so
DANN and CDAN begin their dropout streams from the same seed. These execution constants,
the code hash, and the environment are included in the cross-run experiment lock.
