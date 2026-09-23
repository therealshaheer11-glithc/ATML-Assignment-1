# Task 2 run history and report evidence map

Updated 23 September 2026 from the published code, dated decision records, and the
student's Colab outputs in the working conversation. Final Drive files are still pending
export. Numbers transcribed below are not substitutes for those files.

## Chronology

| Stage | What happened | Disposition |
| --- | --- | --- |
| Original attempt | Six methods/strengths evaluated; DAN λ=10 collapsed, DANN/CDAN had instability. Bandwidth calculation removed zero distances. | Preserved on `archive/task2-attempt-1-20260923`; excluded from the corrected comparison. Its target results were already known. |
| Exploratory Kaggle run | A different λ=10 implementation behaved better. An attempted matched rerun stopped at the runtime/version gate before training. | Neither result replaces the official λ=10 run or isolates a cause of failure. |
| Corrected V1 | D1–D15 agreed, same source split reused, TA zero-distance rule adopted. Source-only and DAN λ=0.1 trained successfully; DAN λ=1 collapsed on source validation. | Diagnostic runs retained in Drive, excluded from official tables. |
| Clipped V2 | TA permitted documented stabilization; global L2 clipping at 20 approved for all six configurations. DAN λ=1 still failed (best source F1 0.1161). | Diagnostic pilot retained; clipping retained in later versions. |
| Normalized-MMD V3 | Approved per-example L2 normalization before MMD, with raw features retained for classification. Source-only and all DAN strengths rerun under the common clipped protocol. DAN λ=1 completed after resuming from epoch 2. | Official Source-only and DAN checkpoints. λ=10 still failed and remains in the study. |
| V3 DANN | Best source F1 0.6049; extreme epoch-average gradient norms and frequent clipping. | Superseded stability pilot, retained separately. |
| Adversarial-normalized V4 | Approved normalized feature input to DANN/CDAN's discriminator branch. DANN reached source F1 0.9409 and predicted all seven source classes. Adoption for DANN/CDAN was recorded using source information; CDAN reached 0.9441. | Official DANN and CDAN. V3 Source-only/DAN stayed fixed. |
| Freeze | All six selections independently checked on source validation and checkpoint identities locked. | Freeze SHA below; no target-based replacement permitted. |
| Final evaluation V5 | All six frozen checkpoints evaluated on 3,929 Sketch images, plus the fixed binary-domain probe. | Completed according to Colab output; actual small results pending publication. |

Earlier source-only baselines did not all collapse. The recorded corrected V1 baseline
reached source F1 0.9374 and the official V3 baseline reached 0.9426. Do not describe the
history as universal baseline failure or attribute every change solely to zero filtering.

## Decisions and deviations

[D1–D17](DECISIONS.md) remain the detailed historical decision record. Their original
future-tense pilot status describes when the document was written; the table above records
the subsequent execution and reported adoption. Preserve the dated revision documents.

- **D1:** DAN strength study; λ=1 reused in the main comparison.
- **D2–D5:** empirical squared mean-embedding MMD (V-statistic, kernel diagonals included);
  strict-upper-triangle bandwidth median excluding self-distances but retaining zero
  distances between distinct samples; summed three-kernel RBF with denominator `2b`;
  detached median; stop on nonpositive/nonfinite bandwidth. The TA allowed either queried
  RBF convention. These are distinct choices; removing kernel diagonals is not the same
  operation as removing bandwidth self-distances.
- **D6–D11:** deterministic cycling with 235 updates per epoch; common AdamW group and
  defaults; float32; common initialization; fixed maximum-budget GRL schedule; earliest
  strictly best source checkpoint with patience five.
- **D12–D15:** balanced binary-domain probe with train-only scaling; all seven labels in
  macro-F1; explicit augmentation details; epoch-complete resume with matching identities.
- **D8 revision, D16, D17:** global clipping, MMD-branch normalization, and adversarial
  feature-branch normalization respectively. These are documented stabilization additions
  under the TA's permission, not requirements in the original manual.

Additional code conventions worth reporting when relevant:

- `torch.median` takes the lower middle value for an even number of pairwise distances;
  it does not average the middle pair. Kernels are summed, not averaged.
- Gradients pass through per-example normalization. Neither CDAN term is detached.
  Classification and the final domain probe use raw backbone features.
- No epsilon/floor, mixed precision, learning-rate warmup, class weighting, or extra
  discriminator updates were silently added.
- Logged loss, gradient norm, discriminator accuracy, GRL strength, and feature norm
  are epoch averages. A maximum across those logs is a **maximum epoch-average**, not
  the largest individual update. Clipping fractions are fractions of updates.
- CDAN logs the norm of the normalized 512-dimensional feature. Its actual outer-product
  discriminator input has norm equal to the softmax probability vector's L2 norm,
  generally less than one. DANN's discriminator directly receives the unit feature.
- Normalized features can still converge toward similar directions. The very small MMD
  bandwidth and source collapse are observations; they do not establish a unique causal
  mechanism or prove that normalization should prevent collapse.
- Failure-example ranking uses NumPy's default `argsort`. Equal-confidence errors are
  not explicitly sorted stably; the original protocol's canonical tie-order wording is
  stronger than this implementation guarantees. This concerns which illustrative error
  images are listed, not checkpoint selection or any reported aggregate metric. Preserve
  the actual output; no training rerun is needed for this presentation detail.
- Final probe accuracy is from a separately fitted logistic regression on raw frozen
  features. It is not the online adversarial discriminator's training accuracy.

## Freeze and exposure disclosure

Reported identities:

```text
source split:
e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74
common initialization state:
4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3
locked expectation:
7cb1a3988b92c5986d850e65d3dc1a928af9d673d19b53b97dd1d90b78a31fb7
source checkpoint audit:
4567df8de5d2b86d662c8c38017df1d0004b626b461c793d767207422033c1ca
six-run checkpoint freeze:
baacc896c12285216eee120b785475b69c1e3bfcee911c33a759a72c897f15b5
```

The original and exploratory target scores had already been seen. The
[locked expectation](../preregistration/DAN_STRENGTH_EXPECTATION.txt) discloses prior
exposure; it cannot be called a target-naive preregistration of the whole project.
Later changes were approved from corrected-run source diagnostics before the corrected
checkpoints' final target evaluation. Preserve this distinction in the report. Code
separates target labels from training/selection; that alone cannot prove the absence of
all human influence from earlier results.

## Execution incidents

The notebook encountered a missing packaged expectation file, a test-import failure,
stale `shared`/`task2` imports from older extracted directories, and an incorrect
hardcoded initialization-hash expectation. Recovery cells were run; later printed
checks verified package/configuration identities and byte-identical V3/V4 initialization.
The latest notebook is needed to preserve the exact recovery and freeze cells. These
setup failures should not be misreported as new successful training experiments.
The original GPU interruption/fresh restarts belong to the archived attempt; the V3
DAN λ=1 resume belongs to the corrected official history.

## What the student needs for the report

1. Cite the assignment protocol and TA permissions; describe D1–D17 and actual deviations.
2. Explain the chronological source-based revisions and prior target exposure.
3. Use the frozen official four-method and strength-study tables, complete histories,
   source-domain metrics, target confusion/class changes, failure examples, and probe.
4. Compare the preserved expectation with observations; monotonicity and near-chance
   separability were hypotheses, not requirements or established outcomes.
5. Keep the failed λ=10 result visible. Distinguish observations from causal explanations;
   neither these single-seed runs nor the unmatched Kaggle experiment isolate causes.
6. Use the task guides for citations and reproduction details. The student must write
   the report prose and interpretation independently; this file is a technical record.
