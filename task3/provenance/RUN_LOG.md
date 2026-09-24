# Task 3 Execution Log

This log records execution evidence reported from the student's Colab notebook. The
original machine-readable artifacts produced by Colab remain authoritative and will be
copied into the persistent Task 3 provenance directory.

## 24 September 2026 - Block 01: environment preflight

Status: `TASK3_BLOCK_01_PASS`

Confirmed behavior:

- Drive was not mounted by the block.
- No dataset was traversed.
- No model was loaded.
- No Sketch image was accessed.
- The source-only phase remains active.
- The approved protocol, artifact identities, source counts, and training constants
  were recorded in `/content/task3_runtime_preflight.json`.

Observed runtime:

| Component | Value |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| NumPy | 2.1.3 |
| scikit-learn | 1.6.1 |
| CUDA | 12.8 |
| cuDNN | 91900 |
| Device type | CUDA |
| GPU | Tesla T4 |

Conclusion: the Task 3 runtime exactly matches the recorded Task 2 runtime. Block 02
may mount Drive, persist the Block 01 JSON, and inspect the two known Task 2 artifact
files by explicit path. PACS must remain untraversed.

## 24 September 2026 - Block 02: Task 2 artifact verification

Status: `TASK3_BLOCK_02_PASS`

Confirmed behavior:

- Google Drive mounted at `/content/drive`.
- The Block 01 JSON was copied to the persistent Task 3 provenance directory.
- PACS was not traversed.
- No Sketch image was accessed.
- The source-only phase remains active.

Persistent Block 01 record:

```text
/content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/
provenance/runtime_preflight.json
```

Persistent Block 01 record SHA-256:
`cee33ce799805a69eea8c484278fb3ef4d5c51f1d3bc33d6cd7d37033788c4c6`

Verified common initialization:

```text
/content/drive/MyDrive/ATML-PA1/task2_corrected_normalized_v3_20260923/
initialization/resnet18_v1_seed6304_common.pt
```

State-dictionary SHA-256:
`4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3`

Verified ERM checkpoint:

```text
/content/drive/MyDrive/ATML-PA1/task2_corrected_normalized_v3_20260923/
source_only/best.pt
```

Checkpoint-file SHA-256:
`3d28a223e4b97b323cb3a20dcb5b7577af96631f2e6ef1f2bc99d53d85761327`

The checkpoint selected epoch 4, reproduced every recorded source-validation metric,
used the approved source protocol and common initialization, and certified that target
labels were not used.

Conclusion: Block 03 may prepare code and a source-only PACS workspace. It may read the
verified PACS archive and approved split manifest, but it must open and extract only the
approved Photo, Art Painting, and Cartoon image members.

## 24 September 2026 - Block 03: source-only data preparation

Status: `TASK3_BLOCK_03_PASS`

Confirmed behavior:

- The repository was checked out at commit
  `12c9c772579d8fe8d129f6345f37043064c9c60c`.
- The exact Task 2 protocol was verified with SHA-256
  `e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74`.
- A source-only Task 3 protocol was created without target records.
- Every approved source image was opened once for verification.
- No Sketch image was opened or extracted.
- The source-only phase remains active.

Repository checkout:

```text
/content/atml_pa1_task3_source
```

Source-only protocol:

```text
/content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/
source_protocol/pacs_sources_seed6304.json
```

Source-only protocol SHA-256:
`626d8517b44ad50c0219adf49e827de6538561386791bed29a9153a589cd6abc`

Verified PACS archive:

```text
/content/drive/MyDrive/ATML-PA1/datasets/PACS_dassl.zip
```

Archive SHA-256:
`0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102`

The archive contains 10,044 members. Only the approved source-domain members were
extracted into:

```text
/content/task3_pacs_sources_v1
```

The source snapshot contains exactly 6,062 images:

| Domain | Image count |
|---|---:|
| Photo | 1,670 |
| Art Painting | 2,048 |
| Cartoon | 2,344 |

Source snapshot SHA-256:
`8ded350769ee15739f8420e755e50ff4377068a4f54ab1c0ba39d5b125e658d2`

Persistent preparation record:

```text
/content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/
provenance/source_data_preparation.json
```

Conclusion: source-only data preparation is complete. The next block may install and
test the approved Task 3 implementation against this exact snapshot. Training must not
start until the code preflight and unit tests pass, and Sketch remains embargoed.

## 24 September 2026 - Block 04 attempt 1: stopped on provenance-field mismatch

Status: stopped before implementation tests or training

The first Block 04 attempt rejected the passing Block 03 record because Block 04
looked for the Block 02 field name `sketch_images_accessed`. Block 03 intentionally
uses the more specific field name `sketch_images_opened_or_extracted`, whose recorded
value is zero. The stop therefore indicates a validation-schema mismatch in Block 04,
not evidence that Sketch was accessed.

The correction makes the prior-record validator specify the expected zero-count field
for each block:

- Block 02: `sketch_images_accessed`;
- Block 03: `sketch_images_opened_or_extracted`.

The failed attempt occurred before unit tests, code preflight, model construction, or
training. It did not open any dataset image and did not access Sketch. Rerunning the
corrected Block 04 is required before training may begin.

## 24 September 2026 - Block 04: implementation verification

Status: `TASK3_BLOCK_04_PASS`

Confirmed behavior:

- Repository commit:
  `19208b4c62acb980fb3246f30e062784b90d8dfc`.
- All 14 target-free unit tests passed.
- The exact locked Colab runtime was preserved.
- The four approved run configurations passed validation.
- Executable code-tree SHA-256:
  `4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44`.
- Locked shared MMD SHA-256:
  `cfe0b1d9c22d7f492ea5e8f76732fbabf21c86cb53f24759af65fda09f9bfbcc`.
- Source-only protocol SHA-256:
  `626d8517b44ad50c0219adf49e827de6538561386791bed29a9153a589cd6abc`.
- Source snapshot SHA-256:
  `8ded350769ee15739f8420e755e50ff4377068a4f54ab1c0ba39d5b125e658d2`.
- Common-initialization state SHA-256:
  `4d53e76c2d8f557b050a1913257c980846bebf6d5b4a28ff4d7cfa12c1d2eef3`.
- ERM checkpoint SHA-256:
  `3d28a223e4b97b323cb3a20dcb5b7577af96631f2e6ef1f2bc99d53d85761327`.
- ERM selected epoch and every source-validation metric were reproduced from the
  checkpoint record.
- Training remained unstarted.
- Sketch images accessed: zero.

Persistent code-preflight record:

```text
/content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/
provenance/code_preflight.json
```

Code-preflight record SHA-256:
`40ef37d7ae08ece5526e5588e446e5301158cc4bd444e2213cedfc0a9bf73eee`

Persistent implementation-verification record:

```text
/content/drive/MyDrive/ATML-PA1/task3_domain_generalization_20260924/
provenance/implementation_verification.json
```

Conclusion: the source-only implementation gate is satisfied. The first approved
training run may begin. Sketch remains embargoed until every run, diagnostic, and final
experiment-lock step is complete.

## 24 September 2026 - Block 05: main DAN-DG lambda 1

Status: `TASK3_BLOCK_05_DAN_DG_1_PASS`, preserved for source-only diagnosis

The prescribed main DAN-DG run completed normally with return code zero and passed the
artifact audit. It stopped after eight epochs because epoch 3 remained the strictly
best source-validation checkpoint and the next five epochs did not improve it.

Selected result:

- selected epoch: 3;
- mean source-validation accuracy: `0.8736406224422547`;
- mean source-validation macro-F1: `0.8693286334613551`;
- worst-source accuracy: `0.8048780487804879`;
- worst-source macro-F1: `0.8051965091232748`;
- selected checkpoint SHA-256:
  `a44bff8e519134459c2d6bf056785801f9944945d160e516b4da7f5e2752992b`;
- history SHA-256:
  `321e267d9b628e357811da617fc5dac07149bdcbd30db030f4c8b65c20518ed7`;
- run-manifest SHA-256:
  `33eb66823509417c758095c820efebdce2b8d90b64a3eecde64e3eaecea83e2f`;
- Sketch images accessed: zero.

The run triggered the approved instability review rather than automatic continuation:

- the clipped-step fraction was `0.9532` in epoch 1 and `1.0` in every later epoch;
- the average pre-clipping gradient norm rose from `156.746653` in epoch 1 to
  `782.310502` in epoch 8;
- mean source-validation macro-F1 fell from the selected `0.869329` to `0.573103` at
  epoch 6 before partially recovering; and
- the selected mean source macro-F1 is materially below the locked ERM value
  `0.9426262342459099`.

No setting is changed automatically. The checkpoint and full history remain valid
evidence for the prescribed main setting. Before another run, inspect the recorded
per-pair bandwidth medians, per-pair MMD values, off-diagonal zero counts, and
post-clipping norms using source-only artifacts. Sketch remains embargoed.

### Source-only instability diagnosis

The detailed diagnostic completed without loading a model or any image. It found:

- all three pairwise bandwidth medians remained positive and finite;
- all three pairs recorded zero off-diagonal zero distances throughout;
- no one domain pair behaved anomalously relative to the other two;
- the medians contracted uniformly, with final-to-initial ratios of
  `0.0014583658` (Photo/Art Painting), `0.0013578689` (Photo/Cartoon), and
  `0.0015334092` (Art Painting/Cartoon);
- the mean MMD remained near `0.31` to `0.37` while distances contracted because the
  median-derived RBF bandwidth contracted with them; and
- Task 2 DAN lambda 1 showed the same qualitative pattern of shrinking bandwidths,
  rising gradients, and increasing clipping, although DAN-DG reached stronger clipping
  sooner and selected a weaker source checkpoint.

Interpretation: this is not evidence of a wrong normalization, excluded-zero error,
pair-order bug, failed clipping operation, or non-finite numerical computation. It is
the scale behavior of the locked adaptive-bandwidth MMD under strong alignment: as the
feature directions contract, the bandwidth follows the distances, the dimensionless
kernel discrepancy need not fall proportionally, and its feature gradients can grow.
The prescribed clipping bound contained every actual update, and source-only early
stopping selected the best permitted checkpoint.

The student subsequently reviewed the explanation and explicitly approved all five
recommendations below:

1. run DAN-DG lambda 0.1 next with every other setting unchanged;
2. retain global gradient clipping at max-norm 20 throughout the lambda study;
3. run lambda 10 only after the lambda 0.1 result has been completed and reviewed,
   even though stronger instability is expected;
4. preserve lambda 1 as the immutable assignment-prescribed main result rather than
   modifying or rerunning it; and
5. postpone any modified-bandwidth exploratory experiment until every required
   DAN-DG and SAM run is complete.

Rationale: lambda 0.1 is not a post-hoc repair or an unauthorized replacement. It is
the next preregistered value in the assignment's bounded DAN-DG strength study. Keeping
the code, initialization, sampling, optimizer, clipping, and checkpoint rule fixed
isolates alignment strength. The lambda 1 result remains the main comparison regardless
of later source or Sketch performance. Lambda 0.1 must be reviewed before lambda 10 is
started; no run is launched automatically. Sketch remains embargoed.

## 24 September 2026 - Block 06: DAN-DG lambda 0.1 controlled study

Status: `TASK3_BLOCK_06_DAN_DG_0P1_PASS`, reviewed before lambda 10

All pre-training gates passed. The run used the same repository commit, executable code
tree, common initialization, source sampling, optimizer, clipping bound, training
budget, and source-only checkpoint-selection rule as lambda 1. Only `lambda_DG` changed
from `1` to the preregistered value `0.1`. It completed ten epochs and stopped after
epochs 6 through 10 failed to improve upon epoch 5.

Selected result:

- selected epoch: 5;
- mean source-validation accuracy: `0.9478614148270105`;
- mean source-validation macro-F1: `0.9462297763360951`;
- worst-source accuracy: `0.926829268292683`;
- worst-source macro-F1: `0.9230060448010302`;
- selected checkpoint SHA-256:
  `dc6036a28e3af8c281b143adb6f47b4d03676fa44c117e912ef9bcd959d8ca27`;
- history SHA-256:
  `fc92efd81ea529a48373977f9d4973e754eb53cfca7622c8513ad5cfc64b35b8`;
- run-manifest SHA-256:
  `731f07bded36f6034e014eb217dca2b1b3a9c7836ed0e5bd72d0c2be3537672a`;
- authorization-record SHA-256:
  `115e50f674a75660d135d9b4d9897879b15538d83ae85521f58c9206ff1e027c`;
- Sketch images accessed: zero.

Per-domain selected source validation:

| Domain | Accuracy | Macro-F1 |
|---|---:|---:|
| Photo | `0.9700598802395209` | `0.9653057911899712` |
| Art Painting | `0.926829268292683` | `0.9230060448010302` |
| Cartoon | `0.9466950959488273` | `0.9503774930172837` |

Source-only review:

- relative to the locked ERM checkpoint, mean macro-F1 increased by
  `0.0036035420901852` and mean accuracy increased by `0.0068252490806001`;
- relative to the prescribed lambda 1 checkpoint, mean macro-F1 increased by
  `0.07690114287474`;
- mean pre-clipping gradient norm stayed between `6.050556` and `11.895829`, compared
  with `156.746653` to `782.310502` for lambda 1;
- only 70 of 2,350 updates were clipped, an aggregate fraction of approximately
  `0.029787`, compared with 1,869 of 1,880 updates, approximately `0.994149`, for
  lambda 1;
- no non-finite loss, invalid bandwidth, failed artifact check, or source-side collapse
  was reported; and
- the MMD value remained scale-adaptive, but multiplying it by 0.1 greatly reduced its
  optimization pressure and allowed the classification objective to remain effective.

Interpretation: this is strong source-only evidence that the lambda 1 behavior is an
alignment-strength effect rather than a broken data pipeline or MMD implementation.
It does not establish that lambda 0.1 generalizes better to Sketch, and it does not
replace lambda 1 as the required main result. The previously approved prerequisite for
lambda 10 is now satisfied: lambda 0.1 completed, was audited, and was reviewed without
Sketch access. Lambda 10 may be launched as the final preregistered DAN-DG strength
condition, with the existing non-finite and early-stopping safeguards unchanged.

## 24 September 2026 - Block 07: DAN-DG lambda 10 controlled study

Status: `TASK3_BLOCK_07_DAN_DG_10_PASS`, preserved source-collapse evidence

All pre-training gates passed, including authentication of the completed and reviewed
lambda 0.1 prerequisite. The run used the same frozen implementation, initialization,
data order, optimizer, clipping rule, training budget, and source-only selection rule
as lambda 0.1 and lambda 1. Only `lambda_DG` changed to the preregistered value `10`.
It completed 12 epochs and selected epoch 7; the strict-improvement rule did not treat
the epoch 8 tie as a new best, and the run stopped after five subsequent stale epochs.

Selected result:

- selected epoch: 7;
- mean source-validation accuracy: `0.21656837139595686`;
- mean source-validation macro-F1: `0.05066996495567924`;
- worst-source accuracy: `0.17270788912579957`;
- worst-source macro-F1: `0.04207792207792208`;
- selected checkpoint SHA-256:
  `f8cc723dc16b5e17a48f8454541beb473b70e6c38d45895f170c09f59821b2c8`;
- history SHA-256:
  `dec8773f2a0642b2aecea85619dbd8bdc19a74723c06a5ce9ca24ce9c8b4b918`;
- run-manifest SHA-256:
  `962626d182952f068a689bb213474074f8f07869c9a5a254cc246164d8fe4330`;
- authorization-record SHA-256:
  `e2d597798af44932d084565430a03010e3cf1cc03e12d5878dd8c5974ca4cbd2`;
- Sketch images accessed: zero.

Per-domain selected source validation:

| Domain | Accuracy | Macro-F1 |
|---|---:|---:|
| Photo | `0.25748502994011974` | `0.058503401360544216` |
| Art Painting | `0.21951219512195122` | `0.05142857142857143` |
| Cartoon | `0.17270788912579957` | `0.04207792207792208` |

Source-only review:

- mean source macro-F1 was `0.8919562692902307` below ERM,
  `0.8186586685056759` below lambda 1, and `0.8955598113804159` below lambda 0.1;
- mean source accuracy was `0.7244677943504535` below ERM;
- all 2,820 updates were clipped;
- mean pre-clipping gradient norm ranged from `1860.699607` to `5171.256474`;
- classification loss ranged from `2.717444` to `4.025770`, always above
  `ln(7)`, approximately `1.94591`, the cross-entropy of uniform seven-class
  predictions;
- the mean MMD remained between approximately `0.36` and `0.42` despite the extreme
  optimization pressure; and
- no non-finite loss, artifact-integrity failure, or Sketch access occurred.

Interpretation: lambda 10 produced finite but severe source-task collapse. This is the
expected excessive-alignment end of the preregistered strength study and is a valid
result, not a failed execution. Together, the three source-only conditions show a clear
strength trade-off: lambda 0.1 preserved classification, lambda 1 destabilized it, and
lambda 10 overwhelmed it. No lambda is promoted or replaced using these observations,
and no conclusion about Sketch is made. The bounded DAN-DG training study is complete.
The next required training run is the independently specified SAM main comparison.

## 25 September 2026 - Block 08: SAM main comparison

Status: `TASK3_BLOCK_08_SAM_PASS`, reviewed before source-only diagnostics

All pre-training gates passed, including authentication of the complete DAN-DG study.
The prescribed standard non-adaptive SAM run used `rho=0.05`, the source ERM
classification objective, two forward/backward passes per update, no clipping of the
first gradient used to construct the ascent perturbation, max-norm 20 clipping only on
the second update gradient, and frozen BatchNorm running statistics during both passes.
It completed ten epochs and stopped after epochs 6 through 10 failed to improve upon
epoch 5.

Selected result:

- selected epoch: 5;
- mean source-validation accuracy: `0.9535960789430552`;
- mean source-validation macro-F1: `0.9540473940355131`;
- worst-source accuracy: `0.9243902439024391`;
- worst-source macro-F1: `0.9236855934108259`;
- selected checkpoint SHA-256:
  `040a80ce15974d71f3a90b5aab5050965d27212822079e5a075c742052c0827d`;
- history SHA-256:
  `d3150a25d014de91c8f349a35dc17ecbac30381920360e0a0438977bf2d1448e`;
- run-manifest SHA-256:
  `d2a1a76200c6af0aed54c5f079d4619be2ab1ddf11c8c723c7f7f2ad12e9dd5e`;
- authorization-record SHA-256:
  `6c399fed3573d88c61895b743c494edd08c2c1bde0b97b107f3e6310893f8e65`;
- Sketch images accessed: zero.

Per-domain selected source validation:

| Domain | Accuracy | Macro-F1 |
|---|---:|---:|
| Photo | `0.9790419161676647` | `0.9757921211248675` |
| Art Painting | `0.9243902439024391` | `0.9236855934108259` |
| Cartoon | `0.9573560767590619` | `0.9626644675708457` |

Source-only review:

- mean source macro-F1 exceeded ERM by `0.0114211597896032` and DAN-DG lambda 0.1
  by `0.0078176176994180`;
- mean source accuracy exceeded ERM by `0.0125599131966448`;
- worst-source macro-F1 exceeded ERM by `0.0117684858172263`;
- the perturbation norm was exactly `0.05` in every epoch;
- perturbed classification loss exceeded base classification loss in every epoch, as
  expected for the normalized ascent step;
- only 5 of 2,350 second-pass update gradients were clipped, an aggregate fraction of
  approximately `0.002128`;
- the mean first-pass gradient norm declined from `5.444643` to `0.893475`, and the
  mean second-pass update-gradient norm remained finite between `3.452772` and
  `7.728617`; and
- no restoration failure, non-finite value, artifact-integrity failure, or Sketch
  access occurred.

Interpretation: SAM trained stably and produced the strongest source-validation
checkpoint among the locked ERM, prescribed DAN-DG lambda 1, and SAM main comparison.
This source-side result does not establish either lower common sharpness or better
Sketch generalization. Both must be measured using the already approved fixed
diagnostics, with Sketch remaining embargoed until the final experiment lock.

### Source-diagnostic subset implementation approval

Before installing or running the source diagnostics, the student explicitly approved:

1. independent NumPy `default_rng(6304)` PCG64 streams for the domain probe and
   sharpness batch, preventing one diagnostic's selection procedure from affecting the
   other's selected examples;
2. fixed Photo, Art Painting, Cartoon processing order, removing ambiguity from the
   order-dependent seeded draws; and
3. sorting selected original validation indices within each domain solely to stabilize
   extraction order and keep identifiers, labels, and feature rows aligned.

These rules do not add class rebalancing and do not change the approved sample counts.
The probe still uses 334 validation examples per source domain, and the sharpness batch
still uses 32 per source domain. Both selections remain source-only, are saved before
feature extraction, and report zero Sketch access.
