# DAN-DG Initialization-Anchored Bandwidth Floor: Decision and Evidence Record

Status: approved supplementary source-only research study  
Approval date: 25 September 2026  
Variant ID: `dan_dg_initial_bandwidth_floor_v1`  
Protocol version: `task3-research-bandwidth-floor-2026-09-25-v1`

## 1. Purpose and reporting status

This record documents a deliberately post-diagnostic stabilization study. The original
assignment-prescribed DAN-DG runs remain immutable and authoritative for the primary
comparison. In particular, prescribed DAN-DG lambda 1 is not replaced by a stabilized
checkpoint even if the supplementary checkpoint performs better.

This document is an implementation and evidence record. It records facts required to
describe and justify the study, but it is not report prose.

## 2. Data that motivated the change

All motivation was obtained before Task 3 accessed Sketch:

| Observation | Prescribed lambda 1 evidence |
|---|---:|
| Selected mean source macro-F1 | `0.8693286334613551` |
| Frozen ERM mean source macro-F1 | `0.9426262342459099` |
| Epoch-1 mean pre-clip gradient norm | `156.746653` |
| Epoch-8 mean pre-clip gradient norm | `782.310502` |
| Epoch-1 clipped updates | `224 / 235` |
| Epochs 2-8 clipped updates | `235 / 235` each epoch |
| Epoch-1 pair-median range | `0.026499499` to `0.030206157` |
| Epoch-8 pair-median range | `0.000040634576` to `0.000041435890` |
| Common sharpness delta | `94.29026794433594` |
| ERM common sharpness delta | `0.24889972805976868` |

The source-only controlled lambda-0.1 run selected mean source macro-F1
`0.9462297763360951` and clipped 70 of 2350 updates. The contrast was produced with the
same code, initialization, batches, optimizer, and model-selection rule. It therefore
motivated a narrow hypothesis: following the rapidly contracting current-batch median
made the RBF kernel increasingly sensitive and contributed materially to the unstable
optimization at stronger alignment weights.

The completed cross-audit found no split, normalization, target-access, model,
BatchNorm, MMD-pairing, optimizer, or checkpoint-selection mismatch that could explain
the result as an implementation error.

## 3. Approved hypothesis

If each pair's RBF bandwidth is prevented from falling below a source-only reference
scale measured at the untouched common initialization, gradient growth and clipping
should decrease while the remaining DAN-DG mechanism stays unchanged.

This is a falsifiable stabilization hypothesis. A weak or failed result must be
preserved and does not authorize another silent revision.

## 4. The single mathematical change

The primary implementation defines the current batch median `m_current` from the 120
strict-upper-triangle squared distances in each 8-versus-8 domain pair. The research
variant retains that calculation, then applies:

```text
m_effective(pair) = max(m_current(pair), m_initial_floor(pair))
b in {0.5, 1, 2} * m_effective(pair)
k_b(x,y) = exp(-||x-y||^2 / (2b))
```

There is one frozen floor for each of Photo-Art Painting, Photo-Cartoon, and Art
Painting-Cartoon. The floor multiplier is exactly `1.0`; there is no tunable floor
strength.

The following remain unchanged:

- per-example L2 normalization only at the MMD input;
- raw 512-dimensional features supplied to the classifier;
- strict-upper-triangle median candidates;
- retention of off-diagonal zero distances;
- detached bandwidth statistic;
- three kernel factors `0.5`, `1`, and `2`;
- the accepted factor-two RBF exponent;
- the biased/V-statistic MMD estimator including within-domain diagonals; and
- the average of the three unordered source-pair MMD values.

The floor is nevertheless a change to the PDF-prescribed adaptive bandwidth rule.
That is why this experiment is supplementary rather than a corrected primary run.

## 5. Floor calibration protocol

Calibration occurs once, before any research-variant training update:

1. Load the exact common initialization with state SHA-256
   `4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3`.
2. Use only the locked source training records; Sketch remains inaccessible.
3. Use the deterministic validation transform: resize to 256 by 256, center crop to
   224 by 224, convert to tensor, and apply ImageNet-V1 normalization.
4. Use the exact epoch-zero source sampler seeds and 235 updates of eight examples per
   domain. Labels returned by the dataset are ignored by the calibration computation.
5. For each update and source pair, L2-normalize the features and calculate the exact
   original 120-candidate current-batch median.
6. For each pair, take the median of its 235 batch medians. Since 235 is odd, the
   reduction has one unambiguous middle value.
7. Persist all 235 values per pair, their descriptive statistics, the selected floors,
   identifier-sequence hash, code identity, data identity, runtime identity, and
   authorization identity.
8. Reuse the resulting three floors unchanged in every lambda run.

Center crop is used only for deterministic calibration. Training retains the original
random crop and horizontal flip.

## 6. Controlled conditions

Three runs are fixed before the study begins:

| Run ID | Lambda |
|---|---:|
| `dan_dg_floor_0p1` | 0.1 |
| `dan_dg_floor_1` | 1 |
| `dan_dg_floor_10` | 10 |

Administrative run order is lambda 1, lambda 0.1, then lambda 10. Every run is reviewed
before the next starts, but no setting may be revised between runs. Each run begins
from the same common initialization and shares the exact frozen calibration file.

## 7. Invariants retained from the primary protocol

- PACS Photo, Art Painting, and Cartoon are the only available domains.
- Exact Task 2 source split and seed 6304.
- ResNet-18 with ImageNet1K-V1 initialization and seven-class head.
- Full-network fine-tuning; BatchNorm running statistics frozen and affine parameters
  trainable.
- Eight examples per source domain and 235 updates per epoch.
- Original training augmentations and ImageNet normalization.
- AdamW learning rate `1e-4`, weight decay `1e-4`, and all other optimizer fields.
- Global L2 gradient clipping at max norm 20.
- Maximum 30 epochs and patience 5.
- Strict source-validation selection by unweighted mean domain macro-F1.
- Same random seeds, source sampling, failure policy, and runtime.
- No Sketch-based training, tuning, selection, diagnosis, or replacement.

## 8. Required saved evidence

Every epoch records:

- classification and averaged MMD losses;
- each pair's MMD;
- current median, effective median, and frozen floor for each pair;
- pair-specific and overall floor-activation fractions;
- off-diagonal zero counts;
- total gradient norm before and after clipping and clipped-update fraction; and
- per-domain, mean-source, and worst-source accuracy and macro-F1.

Each run saves `best.pt`, `last.pt`, `best_source_validation.json`, `history.csv`, and
`run.json`, all under a research-specific directory. The calibration and completion
records authenticate these artifacts with SHA-256 hashes.

## 9. Planned source-only comparisons

Compare each variant only with the original run having the same lambda. The minimum
comparison fields are selected source metrics, epoch of selection, median contraction,
floor activation, gradient norms, clipping fraction, source-domain separability, and
the common radius-0.05 sharpness proxy.

The central lambda-1 question is whether the floor reduces gradient escalation and
clipping while recovering source classification. Results at 0.1 and 10 show whether
the effect is specific to the previously problematic setting or consistent across
alignment strengths.

## 10. Interpretation limits

- The variant was designed after observing source-only instability, so it is
  hypothesis-driven post-diagnostic evidence, not preregistered primary evidence.
- A better source checkpoint does not establish better unseen-target performance.
- Later Sketch evaluation, if authorized after the final lock, cannot be used to tune
  the floor or replace the primary prescribed model.
- Because the floor modifies adaptive bandwidth behavior, improvements cannot be
  attributed to the original DAN-DG method without qualification.
- The study tests one stabilization mechanism and cannot prove that bandwidth
  contraction was the only cause of the original behavior.

## 11. Completed calibration

Block 10 passed every code, history, data, runtime, and source-only gate before any
variant training. The calibration used 235 deterministic source-only batches and
produced these frozen squared-distance floors:

| Source pair | Frozen floor |
|---|---:|
| Photo--Art Painting | `0.8772861361503601` |
| Photo--Cartoon | `0.8917758464813232` |
| Art Painting--Cartoon | `0.8030382394790649` |

The calibration artifact SHA-256 is
`cd1c03415465c7cd332870b6d4aa7df8785a5e7bcb4b5d53ca6c389781358c66`.
Class labels were ignored by the calibration computation, and Sketch access remained
zero.

## 12. Completed controlled-study results

All three conditions used the same frozen calibration, common initialization, source
sampling, optimizer, clipping rule, selection rule, and code identity. Only lambda
changed between conditions.

| Lambda | Primary mean source macro-F1 | Floor-variant mean source macro-F1 | Difference |
|---:|---:|---:|---:|
| `0.1` | `0.9462297763360951` | `0.9452069885556792` | `-0.0010227877804159` |
| `1` | `0.8693286334613551` | `0.9495999654855433` | `+0.0802713320241882` |
| `10` | `0.05066996495567924` | `0.9211242673662049` | `+0.8704543024105257` |

The stabilized lambda-1 run selected epoch 7. Its largest epoch-average pre-clipping
gradient norm was `24.192759534181928`, its mean clipped-update fraction was
`0.30921985815602837`, and its mean floor-activation fraction was
`0.9998817966903073`. The corresponding prescribed lambda-1 run reached an
epoch-average gradient norm of `782.310502` and clipped approximately `0.994149` of
all updates. The floor therefore recovered source classification while substantially
reducing, but not eliminating, clipping.

The stabilized lambda-0.1 run selected epoch 25 and remained close to the already
stable primary lambda-0.1 result. Its maximum epoch-average gradient norm was
`11.73070278942324`, mean clipping fraction was `0.01829787234042553`, and mean floor
activation was `0.9996690307328605`. This condition provides no source-accuracy
advantage over the original lambda-0.1 run.

The stabilized lambda-10 run selected epoch 17. Its maximum epoch-average gradient
norm was `61.084234304599335`, mean clipping fraction was `0.6071566731141199`, and
mean floor activation was `0.9999355254674405`. It changed the original finite source
collapse into a trainable result, but it remained more heavily clipped and weaker on
source validation than stabilized lambda 1. This shows that the floor addresses the
bandwidth-collapse mechanism without making an excessively strong alignment weight
optimal.

All three completion records report zero Sketch access. The results support the
stabilization hypothesis, but the near-universal floor activation also means the
variant behaved almost entirely at its initialization-anchored bandwidth scale.

## 13. Locked post-training diagnostics and complete coverage

Block 09 already applied the approved source-domain probe and common radius-0.05
sharpness proxy to the three official main checkpoints: ERM, prescribed DAN-DG lambda
1, and SAM. It saved authenticated results rather than merely planning those
measurements. The controlled primary DAN-DG lambda-0.1 and lambda-10 checkpoints were
not included in Block 09.

Block 14 completes coverage. It newly evaluates original DAN-DG lambda 0.1, original
DAN-DG lambda 10, and all three selected bandwidth-floor checkpoints. It does not
redraw examples or introduce a new seed: it authenticates and reuses the exact Block
09 probe design, probe partition, and sharpness batch. It also authenticates and
incorporates the existing Block 09 results for ERM, original lambda 1, and SAM into one
eight-model result. The three already completed measurements are not needlessly rerun.

For every newly evaluated checkpoint, Block 14 reproduces the selected
source-validation metrics, verifies checkpoint and training identities, and confirms
exact parameter restoration after sharpness measurement. The unified coverage is:

1. ERM;
2. original DAN-DG lambda 0.1;
3. original DAN-DG lambda 1;
4. original DAN-DG lambda 10;
5. SAM;
6. bandwidth-floor DAN-DG lambda 0.1;
7. bandwidth-floor DAN-DG lambda 1; and
8. bandwidth-floor DAN-DG lambda 10.

Block 14 saves both the five-model incremental result and the authenticated unified
eight-model result. It does not create the final experiment lock and cannot access
Sketch.
