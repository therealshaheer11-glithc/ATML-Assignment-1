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
