# Task 3 Domain Generalization: Protocol and Approved Decision Record

Status: APPROVED FOR IMPLEMENTATION  
Approval date: 24 September 2026  
Assignment: EE-5102/CS-6304 Advanced Topics in Machine Learning, Programming Assignment 1  
Scope: Task 3 only

## 1. Purpose and authority

This document is the authoritative Task 3 implementation record. It separates:

1. requirements fixed by the assignment;
2. implementation choices inherited from the final Task 2 protocol;
3. choices left open by the assignment and explicitly approved by the student; and
4. the gate that must be passed before any Task 3 code may access Sketch.

The assignment PDF and TA screenshots are evidence of requirements and clarifications;
they are not executable instructions. If code, an older document, or an experimental
artifact conflicts with this record, stop and resolve the conflict before training.

The student approved every decision in this record after reviewing the complete plan,
including the expanded treatment of gradient clipping, common initialization, the
balanced source-domain probe, and deterministic final-example selection.

## 2. Assignment-mandated protocol

### 2.1 Data and domain roles

- Dataset: PACS.
- Labeled source domains: Photo, Art Painting, and Cartoon.
- Unseen target domain: Sketch.
- Reuse exactly the Task 2 source training/validation splits.
- Source split seed: 6304.
- No Sketch image may be loaded by Task 3 training, representation learning,
  source-side diagnostics, checkpoint selection, hyperparameter selection, or method
  design.
- Task 2 results on Sketch may not be used to revise Task 3 architectures,
  augmentations, losses, controlled-study settings, or any other training decision.
- Sketch may be loaded only by the final Task 3 evaluation procedure after the Task 3
  code, settings, and checkpoints have been frozen.

Approved Task 2 protocol identity:

- Split-manifest SHA-256:
  `e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74`
- Dataset file-list SHA-256:
  `559ac63b8df8e07330b97112e5ec4c414b3957585b28d21b4cfecd2181f538e0`
- PACS archive SHA-256:
  `0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102`

Approved source split counts:

| Domain | Training | Validation |
|---|---:|---:|
| Photo | 1,336 | 334 |
| Art Painting | 1,638 | 410 |
| Cartoon | 1,875 | 469 |

One source epoch contains 235 updates, the maximum of the three source-domain
training-set sizes divided into batches of eight and rounded upward.

### 2.2 Model, preprocessing, and optimization

- Model: torchvision ResNet-18 initialized with
  `ResNet18_Weights.IMAGENET1K_V1`.
- Replace the ImageNet classifier with a seven-class linear classifier.
- Fine-tune the complete network for every newly trained Task 3 method.
- Feature used by MMD and diagnostics: the 512-dimensional representation immediately
  before the classifier.
- Decode images with Pillow and convert them to RGB.
- Resize directly to 256 by 256 using bilinear interpolation with antialiasing.
- Training augmentation: random 224 by 224 crop without padding, followed by a
  horizontal flip with probability 0.5.
- Validation and evaluation transformation: 224 by 224 center crop with no flip.
- Apply the normalization supplied by the ImageNet-V1 pretrained weights.
- Each training update contains exactly eight examples from each source domain.
- Use domain-balanced batches in fixed concatenation order: Photo, Art Painting,
  Cartoon.
- Train for at most 30 source epochs.
- Use AdamW with learning rate `1e-4` and weight decay `1e-4`.
- Select checkpoints using the unweighted mean macro-F1 across the three source
  validation domains.
- Stop after five complete epochs without strict improvement.
- Use seed 6304.
- Freeze every BatchNorm running mean, running variance, and batch counter at its
  pretrained ImageNet value during every training pass.
- BatchNorm scale and bias parameters remain trainable.

### 2.3 Required methods

1. ERM: reuse the saved Task 2 Source-only checkpoint without retraining it.
2. DAN-DG: train ERM plus the average pairwise source-domain MMD penalty.
3. SAM: train ERM using standard, non-adaptive Sharpness-Aware Minimization.

Required main settings:

- DAN-DG: `lambda_DG = 1`.
- SAM: `rho = 0.05`.
- AdamW learning rate and weight decay remain identical to ERM for SAM.

### 2.4 Required reporting evidence

- Accuracy and macro-F1 for each source validation domain.
- Mean-source and worst-source accuracy and macro-F1.
- Final Sketch accuracy and macro-F1.
- Change in Sketch accuracy relative to ERM.
- Source-domain separability for ERM, DAN-DG, and SAM.
- The common local sharpness proxy for ERM, DAN-DG, and SAM.
- Classification-loss curves for every trained method.
- MMD-penalty curves for DAN-DG.
- A compact table or plot for the controlled study.
- Per-class Sketch changes, important confusions, and selected failures.
- A final comparison with the corresponding Task 2 results.

## 3. Normalization rules: keep the two Task 2 rules distinct

Task 2 used two different normalization rules. They must never be conflated.

### 3.1 DAN MMD-input normalization - inherited by Task 3

For DAN and DAN-DG only, L2-normalize each 512-dimensional feature immediately before
the MMD calculation:

```text
f_mmd = f / ||f||_2
```

The classifier continues to receive the original, unnormalized feature `f`. No epsilon
or clamp is introduced. A zero or non-finite feature norm stops the run.

Task 3 DAN-DG must use this rule because the assignment requires the same MMD
implementation and kernel construction as Task 2.

### 3.2 DANN/CDAN discriminator-input normalization - not used by Task 3

The separate Task 2 adversarial rule normalized features immediately before the DANN
or CDAN domain discriminator while leaving classifier features unnormalized. Task 3
contains neither DANN nor CDAN and has no domain discriminator. Therefore this
adversarial-input normalization must not be applied anywhere in Task 3.

## 4. TA MMD clarifications and the locked MMD definition

The TA clarified the following:

1. diagonal self-distances may be removed from the bandwidth-median candidates;
2. zero distances between different examples should remain unless they cause another
   issue; and
3. either of the two presented RBF exponent conventions is acceptable.

The approved Task 2 implementation selected one convention, and Task 3 will reuse it
unchanged.

For each current source-domain pair:

1. L2-normalize the eight features from each domain using the DAN rule in Section 3.1.
2. Concatenate the two domains to obtain 16 features.
3. Compute every pairwise squared Euclidean distance.
4. Form bandwidth candidates from the strict upper triangle, so each distinct pair is
   used once and diagonal self-distances are excluded.
5. Retain all off-diagonal zero distances.
6. Let `m` be the median of those candidates and detach it from autograd.
7. Stop with a diagnostic if `m` is non-positive or non-finite; do not filter
   additional zeros, clamp the bandwidth, or substitute a fallback without new student
   approval.
8. Define bandwidths `b` in `{0.5m, m, 2m}`.
9. Use the locked kernel convention:

```text
k_b(x, y) = exp(-||x-y||^2 / (2b))
```

10. Sum the three RBF kernels.
11. Compute the literal empirical squared mean-embedding distance:

```text
MMD^2 = mean(K_xx) + mean(K_yy) - 2 mean(K_xy)
```

The within-domain kernel diagonals are included in `mean(K_xx)` and `mean(K_yy)`. This
is the same biased/V-statistic estimator used in Task 2.

For the three unordered source pairs `(Photo, Art Painting)`, `(Photo, Cartoon)`, and
`(Art Painting, Cartoon)`, define:

```text
pairwise_mmd = (MMD_PA^2 + MMD_PC^2 + MMD_AC^2) / 3
L_DAN-DG = L_ERM + lambda_DG * pairwise_mmd
```

The main comparison uses `lambda_DG = 1`.

## 5. Approved controlled study and preregistered expectation

The approved controlled study varies:

```text
lambda_DG in {0.1, 1, 10}
```

Every other setting remains fixed. The `lambda_DG = 1` run is reused in the main
comparison. Sketch results from the study are analysis-only and may not replace the
prescribed main setting with a post-hoc winner.

The preregistered expectation is:

- increasing `lambda_DG` should place stronger pressure on the three observed source
  feature distributions to become similar;
- source-domain separability should generally decrease as alignment pressure grows;
- excessive alignment may reduce mean-source or worst-source classification by
  suppressing class-discriminative information; and
- Sketch performance may be non-monotonic, but no Sketch outcome will be used to tune
  or replace the main setting.

This study was selected for conceptual comparability with target-aware DAN, not because
of any Task 2 Sketch result.

## 6. Approved initialization and ERM handling

### 6.1 ERM

Load the exact selected Task 2 Source-only checkpoint. Do not retrain ERM.

- Selected epoch: 4.
- Checkpoint SHA-256:
  `3d28a223e4b97b323cb3a20dcb5b7577af96631f2e6ef1f2bc99d53d85761327`
- Expected mean source-validation macro-F1: `0.9426262342459099`.
- Expected mean source-validation accuracy: `0.9410361657464104`.

Expected per-domain source-validation results:

| Domain | Accuracy | Macro-F1 |
|---|---:|---:|
| Photo | 0.9730538922 | 0.9681339341 |
| Art Painting | 0.9097560976 | 0.9119171076 |
| Cartoon | 0.9402985075 | 0.9478276610 |

These values must be reproduced before accepting the Task 3 evaluation setup.

### 6.2 Newly trained methods

DAN-DG and SAM must both start from the exact common initialization used by Task 2:

- common-initialization state SHA-256:
  `4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3`.

This state contains the ImageNet-V1 ResNet-18 and the seed-6304 initialized seven-class
head before source training. DAN-DG and SAM must not start from the trained ERM
checkpoint, because doing so would give them extra supervised training and invalidate
the controlled comparison.

If the common-initialization artifact is unavailable, reconstruction is permitted only
if its state hash exactly matches the value above. A mismatch stops the workflow.

## 7. Approved optimizer completion and gradient clipping

Retain the Task 2 optimizer completion:

- AdamW `betas=(0.9, 0.999)`;
- `eps=1e-8`;
- `amsgrad=False`;
- one parameter group containing every trainable model parameter;
- constant learning rate;
- no scheduler or warmup;
- float32 without autocast, GradScaler, or gradient accumulation;
- ordinary mean cross-entropy;
- no label smoothing or class weighting; and
- deterministic non-foreach, non-fused optimizer behavior.

Retain global L2 gradient clipping with `max_norm=20`:

- DAN-DG: clip the complete gradient of classification plus weighted MMD immediately
  before the AdamW update.
- SAM first pass: do not clip the gradient used to construct the normalized ascent
  perturbation. This pass performs no optimizer update, and global clipping would only
  rescale the vector that is immediately normalized.
- SAM second pass: after computing the gradient at the perturbed parameters and
  restoring the original parameters, clip the complete gradient to 20 immediately
  before the AdamW update.

For every actual update, log the pre-clipping norm, post-clipping norm, and whether
clipping occurred. Stop on a non-finite loss or gradient.

## 8. Approved SAM implementation

Use standard, non-adaptive SAM with `rho=0.05`.

For each already-created domain-balanced source batch:

1. compute ordinary ERM cross-entropy at `theta`;
2. compute its gradient over every trainable backbone, BatchNorm-affine, and classifier
   parameter;
3. stop if the global gradient norm is zero or non-finite;
4. construct the global perturbation
   `epsilon = 0.05 * gradient / ||gradient||_2`;
5. add the perturbation without changing optimizer state;
6. recompute the same ERM loss on the same batch at `theta + epsilon`;
7. restore every original parameter exactly;
8. clip the second-pass gradient globally to 20; and
9. perform one AdamW update on the original parameters.

BatchNorm running statistics remain frozen during both passes. AdamW weight decay is
applied only by the final optimizer update and is not part of the ascent perturbation.
The implementation must restore parameters in a failure-safe manner and test exact
restoration.

## 9. Approved deterministic data and execution policy

- Each source loader emits batches of eight.
- An epoch has 235 updates.
- Smaller source loaders cycle through fresh deterministic shuffled permutations.
- No example is dropped merely to make an epoch divide evenly.
- Use the same epoch-specific sampler and worker-seed construction for all methods.
- Reuse identical source sampling and augmentation streams across comparable runs.
- Use two data-loader workers and pinned memory on CUDA.
- Disable TF32 and cuDNN benchmarking.
- Enable deterministic algorithms and deterministic cuDNN behavior.
- Save resumable state only after a complete epoch.
- A resumed run must match its configuration, code identity, split identity,
  environment identity, device type, and saved RNG states.
- An interrupted partial epoch is discarded and replayed from the last complete epoch.

The first Task 3 run records the core software and device environment. All subsequent
Task 3 training runs must match that locked environment or start as a separately
documented experiment rather than silently resuming or joining the comparison.

## 10. Source-only data boundary and target embargo

Create a Task 3 source-only protocol derived mechanically from the approved Task 2
source splits. Record its parent Task 2 protocol hash. The Task 3 training and
source-diagnostic paths must:

- contain only Photo, Art Painting, and Cartoon records;
- reject a source record whose first path component is not an approved source domain;
- never instantiate the Task 2 unlabeled-Sketch dataset;
- never traverse or verify the Sketch directory;
- expose no target-loader argument; and
- log that zero Sketch images were accessed.

`evaluate_sketch.py` is the only Task 3 component permitted to instantiate a Sketch
dataset, and it must refuse to run without the final experiment-lock manifest.

## 11. Approved source-domain separability diagnostic

The source-validation sizes are unequal, so construct a domain-balanced probe:

1. use all 334 Photo validation examples;
2. select 334 Art Painting validation examples without replacement using seed 6304;
3. select 334 Cartoon validation examples without replacement using seed 6304;
4. save the resulting 1,002 image identifiers before extracting model features;
5. reuse exactly the same identifiers for ERM, DAN-DG, and SAM;
6. do not introduce additional class rebalancing unless separately approved;
7. extract the raw, unnormalized 512-dimensional features;
8. create one domain-stratified 70/30 train/test partition using seed 6304 and reuse it
   for every model;
9. fit `StandardScaler` on probe-training features only;
10. train multinomial `LogisticRegression` with `C=1`, solver `lbfgs`,
    `max_iter=2000`, and `random_state=6304`; and
11. report ordinary held-out domain accuracy, with chance equal to 33.3 percent.

This design prevents the probe from exploiting unequal domain sample counts and keeps
the compared image set identical across representations.

## 12. Approved common sharpness diagnostic

Before examining model results:

1. select 32 validation examples from each source domain uniformly without replacement
   using seed 6304;
2. save the 96 image identifiers;
3. reuse exactly the same fixed batch for every model; and
4. use the deterministic validation transform.

For each model:

- place the complete model in evaluation mode;
- use ordinary cross-entropy averaged over the 96 examples;
- include every trainable backbone, BatchNorm-affine, and classifier parameter;
- exclude AdamW weight decay and optimizer state from the diagnostic loss;
- compute the global cross-entropy gradient;
- form `epsilon = 0.05 * gradient / ||gradient||_2`;
- measure `delta_sharp = L_val(theta + epsilon) - L_val(theta)`;
- restore the original parameters exactly; and
- verify the restoration before continuing.

This value is a standardized local proxy only. It must not be described as proof that
an entire loss landscape is globally flatter.

### 12.1 Approved deterministic subset implementation

The student explicitly approved the following implementation details before the source
diagnostics were committed or run:

- use one NumPy `default_rng(6304)` PCG64 stream for source-domain-probe subset
  selection;
- use a separate NumPy `default_rng(6304)` PCG64 stream for sharpness-batch subset
  selection, so changes to one diagnostic cannot silently change the other's sample;
- process domains in the fixed order Photo, Art Painting, Cartoon because seeded random
  draws are order-dependent; and
- sort the selected original validation indices within each domain before extraction.

Sorting does not change which examples were randomly selected. It only fixes their
loading order so saved identifiers, labels, and feature rows remain aligned and every
model receives the same examples in the same order. The full selected identifiers and
the probe partition are persisted before model feature extraction.

## 13. Approved checkpoint and metric conventions

- Validate after every complete epoch.
- Replace the selected checkpoint only on strict improvement in unweighted mean source
  macro-F1.
- Exact ties retain the earlier checkpoint and count toward patience.
- There is no epoch-zero checkpoint candidate.
- Validation is deterministic and unshuffled.
- Every macro-F1 call explicitly supplies all seven class IDs and uses
  `zero_division=0`.
- Report worst-source accuracy as the minimum of the three domain accuracies.
- Report worst-source macro-F1 as the minimum of the three domain macro-F1 values.
- Preserve classification loss, average MMD, per-pair MMD, bandwidth diagnostics,
  gradient norms, clipping incidence, and validation metrics in machine-readable
  histories.

## 14. Final experiment lock and Sketch evaluation

Before the first Task 3 Sketch access, create an immutable experiment-lock manifest
containing:

- code-tree hash;
- configuration hashes;
- source-protocol and source-subset hashes;
- common-initialization identity;
- ERM, DAN-DG, and SAM checkpoint hashes;
- selected epochs and source-validation metrics;
- source-domain probe identifiers and partition;
- sharpness-batch identifiers;
- controlled-study preregistration hash;
- runtime environment;
- confirmation that no Sketch image was accessed; and
- explicit student authorization to begin final evaluation.

After the lock, evaluate the complete Sketch domain. Target labels may be used only for
final metrics, per-class analysis, confusion analysis, and example inspection. No
training or selection decision may change in response.

## 15. Approved deterministic final-example selection

Example selection occurs only after the experiment lock and final Sketch evaluation.
For each of DAN-DG and SAM:

1. compute every class's Sketch accuracy change relative to ERM;
2. identify the class with the largest degradation;
3. within that class, identify the dominant incorrect predicted class;
4. select up to two highest-confidence incorrect examples from that confusion;
5. identify the class with the largest improvement;
6. select one example that ERM gets wrong and the method gets right; and
7. break class ties by the fixed class ID and example ties by image path after the
   primary confidence ordering.

If the required category has no eligible example, record that fact rather than
substituting a subjective example. This produces at most three deterministic examples
per newly trained method and prevents cherry-picking while still showing both failure
and improvement behavior.

## 16. Approved instability and change-control policy

No unprescribed stabilization or design change is automatic.

If source-only evidence shows collapse, divergence, invalid bandwidths, non-finite
values, abnormal clipping, missing-class predictions, or another material problem:

1. stop the affected run;
2. preserve its complete artifacts as diagnostic evidence;
3. diagnose using source information only;
4. explain the failure, proposed change, and comparison consequences to the student;
5. obtain explicit student approval;
6. record the revision in a new versioned decision document;
7. use a new output root; and
8. do not inspect Sketch while making or adopting the decision.

The TA's permission to make well-reasoned, explainable changes does not authorize
silent changes, target-guided tuning, or replacement of the assignment's main settings.

## 17. Planned run matrix

| Run | Training required | Main/study role | Initialization |
|---|---|---|---|
| ERM | No | Main baseline | Load selected Task 2 ERM checkpoint |
| DAN-DG 0.1 | Yes | Controlled study | Exact common Task 2 initialization |
| DAN-DG 1 | Yes | Main and controlled study | Exact common Task 2 initialization |
| DAN-DG 10 | Yes | Controlled study | Exact common Task 2 initialization |
| SAM 0.05 | Yes | Main comparison | Exact common Task 2 initialization |

## 18. Planned repository structure

```text
shared/
  mmd.py
  pacs.py
  splits/
    pacs_sources_seed6304.json
task3/
  configs/
    base.json
    dan_dg_0p1.json
    dan_dg_1.json
    dan_dg_10.json
    sam.json
  docs/
    TASK3_PROTOCOL_AND_DECISIONS.md
  preregistration/
    DAN_DG_STRENGTH_EXPECTATION.md
  methods/
    dan_dg.py
    sam.py
  selection/
    source_validation.py
  evaluation/
    source_domain_separability.py
    sharpness.py
    evaluate_sketch.py
    class_analysis.py
  notebook_blocks/
    01_environment_preflight.py
    02_mount_and_verify_task2_artifacts.py
    03_prepare_source_only_data.py
    04_install_and_verify_implementation.py
  tests/
    test_task3_core.py
  train.py
  preflight.py
  freeze.py
  results/
  README.md
```

Existing Task 2 code, results, splits, and evidence are immutable provenance. Task 3
must be implemented in new files and must import or copy only verified shared behavior.

## 18.1 Execution status

### Post-lambda-1 source-only review and authorization

After the prescribed DAN-DG lambda 1 run and its source-only instability diagnosis,
the student explicitly approved the following change-control decisions:

1. DAN-DG lambda 0.1 is the next run, with every other setting unchanged.
2. The global gradient clipping max-norm remains 20 for all three lambda runs.
3. DAN-DG lambda 10 may run only after lambda 0.1 has completed and been reviewed.
4. DAN-DG lambda 1 is immutable as the assignment-prescribed main result and will not
   be modified, overwritten, rerun, or replaced by a study value.
5. Any modified-bandwidth exploratory experiment is postponed until all required
   DAN-DG and SAM runs are complete and would require a new, explicitly labeled
   protocol rather than replacing an official result.

This authorization responds only to source-side evidence. No Sketch image, label,
metric, example, or Task 2 Sketch result informed it. The controlled-study run is an
approved measurement of alignment strength, not a revised main configuration.

### Lambda 0.1 review outcome

DAN-DG lambda 0.1 completed in ten epochs and selected epoch 5 with mean source
macro-F1 `0.9462297763360951` and worst-source macro-F1 `0.9230060448010302`.
Across all 2,350 updates, only 70 were clipped. The run preserved the exact frozen
implementation and reported zero Sketch access. The result passed its artifact audit
and did not exhibit the gradient escalation seen at lambda 1.

This review satisfies the student's condition that lambda 0.1 be completed and
reviewed before lambda 10. The next authorized run is therefore DAN-DG lambda 10 under
the same frozen implementation. Lambda 10 remains an intentionally strong controlled-
study condition; a weak or collapsed source result will be preserved as evidence and
will not be silently stabilized, replaced, or rerun.

### Lambda 10 review outcome

DAN-DG lambda 10 completed in 12 epochs and selected epoch 7 with mean source
macro-F1 `0.05066996495567924` and worst-source macro-F1
`0.04207792207792208`. Every one of its 2,820 updates was clipped, and its mean
pre-clipping gradient norm ranged from `1860.699607` to `5171.256474`. The run
remained finite, passed its artifact audit, and reported zero Sketch access.

The finite result is preserved as severe source-collapse evidence for excessive
alignment. It is not rerun with different clipping, bandwidth, normalization,
optimization, or sampling. The complete preregistered DAN-DG strength study is now
finished. The next required training condition is standard non-adaptive SAM at
`rho=0.05`, using only the ERM classification objective and the frozen common Task 3
settings.

### SAM review outcome

Standard non-adaptive SAM at `rho=0.05` completed in ten epochs and selected epoch 5
with mean source macro-F1 `0.9540473940355131` and worst-source macro-F1
`0.9236855934108259`. Its ascent perturbation norm was exactly `0.05`, only 5 of
2,350 second-pass update gradients were clipped, all values remained finite, and the
artifact audit reported zero Sketch access.

All required source-only training is now complete. No training checkpoint will be
rerun, replaced, or selected again. The next phase is restricted to the approved
source-domain-separability and common-sharpness diagnostics. These diagnostics must be
completed and frozen before any Sketch record is made accessible.

Blocks 01 through 03 completed successfully on 24 September 2026.

- Block 01 locked the runtime: Python 3.13.15, PyTorch 2.11.0+cu128, torchvision
  0.26.0+cu128, NumPy 2.1.3, scikit-learn 1.6.1, CUDA 12.8, cuDNN 91900, and Tesla T4.
- Block 02 persisted the runtime record and verified the common initialization and
  exact Task 2 ERM checkpoint without traversing PACS.
- Block 03 checked out repository commit
  `12c9c772579d8fe8d129f6345f37043064c9c60c`, mechanically derived the source-only
  protocol, and verified all 6,062 approved source images.
- The source-only protocol SHA-256 is
  `626d8517b44ad50c0219adf49e827de6538561386791bed29a9153a589cd6abc`.
- The extracted source snapshot SHA-256 is
  `8ded350769ee15739f8420e755e50ff4377068a4f54ab1c0ba39d5b125e658d2`.
- Every completed block reported zero Sketch images opened or extracted.
- Block 04 subsequently passed all 14 target-free unit tests and the complete code
  preflight at repository commit
  `19208b4c62acb980fb3246f30e062784b90d8dfc`. It locked executable code-tree SHA-256
  `4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44`,
  reported that training had not started, and again recorded zero Sketch access.

The implementation now provides four locked training configurations, source-only data
validation, DAN-DG pairwise MMD, standard non-adaptive SAM, deterministic training and
resume logic, source-validation selection, and a target-free code preflight. The fast
unit-test suite covers the two distinct normalization policies, target-record
rejection, deterministic sampling, the inherited transform pipeline, frozen
BatchNorm, the exact three-pair MMD reduction and bandwidth failure rule, global
gradient clipping, SAM's unclipped first gradient, second-gradient clipping, exact
restoration, and failure-safe restoration. Training remains prohibited until the
implementation is committed, installed in Colab, and passes both the unit tests and
the code preflight.

## 19. Report and repository boundary

- Code assistance is permitted, but the student remains responsible for understanding
  every submitted line.
- Materially reused external code must be attributed in the README.
- The repository must include code, configurations, split logic/indices, environment
  information, machine-readable results, and reproduction instructions.
- Do not commit raw datasets or unnecessary large checkpoints.
- The assignment prohibits generative AI from writing any part of the submitted PDF
  report. Implementation records, code, machine-readable evidence, and factual run
  manifests do not replace student-authored report language or interpretation.
