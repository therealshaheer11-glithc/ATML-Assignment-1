# Task 2 — PACS domain adaptation

Photo, Art Painting, and Cartoon are labeled sources; Sketch is the unlabeled adaptation
target. Checkpoints are selected using source validation and frozen before Sketch-label
evaluation.

**Status:** all six official runs and the frozen-checkpoint evaluation are complete.
The repository includes the official histories, prediction tables, metrics, plots,
freeze, adoption record, and corrected execution notebook. Checkpoints and PACS images
remain outside Git.

## Start here

| Question | Record |
| --- | --- |
| What does the assignment require? | [Assignment protocol](docs/ASSIGNMENT-PROTOCOL.md) |
| Which choices did we approve? | [D1–D17 decisions](docs/DECISIONS.md) |
| What changed, and why? | [Chronological run history and clarifications](docs/RUN-HISTORY.md) |
| Which code produced the experiments? | [Pinned reproduction instructions](docs/REPRODUCTION.md) |
| Where is every attempt/version mapped? | [Complete source lineage](provenance/SOURCE-LINEAGE.md) |
| How were target labels and the domain probe handled? | [Final evaluation protocol](docs/FINAL-EVALUATION-PROTOCOL.md) |
| Where are the published result files? | [Saved-results index](results/README.md) |
| What has been verified? | [Repository audit and handoff](docs/REPOSITORY-AUDIT.md) |

## Official run set

The following values summarize the published machine-readable results in
[`results/final`](results/final/).
Source F1 is the unweighted mean of seven-class macro-F1 across the three source domains.

| Run | Training version | Selected epoch | Source F1 | Sketch accuracy | Sketch F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Source-only | V3 | 4 | 0.9426 | 0.6200 | 0.6574 |
| DAN λ=0.1 | V3 | 2 | 0.9320 | 0.6147 | 0.6409 |
| DAN λ=1 | V3 | 10 | 0.9352 | 0.7582 | 0.7076 |
| DAN λ=10 | V3 | 1 | 0.0507 | 0.0407 | 0.0112 |
| DANN | V4 | 9 | 0.9409 | 0.6908 | 0.7115 |
| CDAN | V4 | 6 | 0.9441 | 0.7203 | 0.7464 |

The main four-method table uses Source-only, DAN λ=1, DANN, and CDAN. The controlled
study uses Source-only and all three DAN strengths. The failed λ=10 run remains included.
V5 names the final evaluator; it is not another training version.

## Common protocol

- ResNet-18 ImageNet V1, a seven-class head, full fine-tuning, fixed common initialization.
- Fixed [seed-6304 source split](../shared/splits/pacs_sketch_seed6304.json):
  Photo 1,336/334, Art Painting 1,638/410, Cartoon 1,875/469 train/validation images.
- Eight images per source domain per update; adaptation adds 24 unlabeled Sketch images.
  One epoch is 235 updates. Source-only does not train on target images.
- AdamW, learning rate and weight decay both `1e-4`; at most 30 epochs;
  patience five on mean source-validation macro-F1, earliest checkpoint on exact ties.
- ImageNet BatchNorm running statistics remain frozen; affine parameters are trainable.
- Global L2 gradient clipping at 20 for every official method. DAN uses per-example
  feature normalization before MMD; V4 DANN/CDAN normalize the feature branch entering
  the discriminator. Classification uses the original feature throughout.
- TA-approved bandwidth handling excludes self-distances and retains off-diagonal zeros.
  The approved estimator, kernel convention, and all further details are in D2–D5.

## Artifacts and storage

Persistent directories under `/content/drive/MyDrive/ATML-PA1/`:

| Directory | Role |
| --- | --- |
| `task2_corrected_20260923` | Corrected, unclipped diagnostic runs |
| `task2_corrected_clipped_v2_20260923` | Clipped diagnostic pilot |
| `task2_corrected_normalized_v3_20260923` | Official Source-only/DAN; superseded DANN pilot |
| `task2_adversarial_normalized_v4_20260923` | Adopted DANN/CDAN and adoption record |
| `task2_official_freeze_20260923` | Six-run freeze and independent source checkpoint audit |
| `task2_final_evaluation_20260923` | Final tables, predictions, probe evidence, and plots |

Keep these directories and the common initialization. The exporter in
[`tools/export_task2_evidence.py`](../tools/export_task2_evidence.py) produced the
published small-file evidence without modifying runs or exporting checkpoints.
The export record and source paths are preserved in
[`FINAL-EVIDENCE-EXPORT.json`](provenance/FINAL-EVIDENCE-EXPORT.json).

## Reproduction and attribution

Use [REPRODUCTION.md](docs/REPRODUCTION.md). Resume requires the exact recorded snapshot
and runtime; do not use this reorganized checkout to resume an old run.
Historical [manifests](provenance/README.md) retain their original bytes and commit context.

Implementation uses PyTorch, torchvision, NumPy, Pillow, scikit-learn, and Matplotlib.
The model uses torchvision's ImageNet-V1 ResNet-18 weights. No public DAN/DANN/CDAN
implementation was copied. Codex assisted with implementation and technical documentation;
the student must independently write the report's prose and interpretation.
