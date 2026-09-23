# Task 2 — PACS domain adaptation

Photo, Art Painting, and Cartoon are labeled sources; Sketch is the unlabeled adaptation
target. Checkpoints are selected using source validation and frozen before Sketch-label
evaluation.

**Status:** the student has reported completion of all six official runs and the final
Colab evaluation. The final prediction tables, plots, histories, freeze, and adoption
record still need to be exported from Drive and published. This repository currently
contains the implementation and protocols, not those final result files.

## Start here

| Question | Record |
| --- | --- |
| What does the assignment require? | [Assignment protocol](docs/ASSIGNMENT-PROTOCOL.md) |
| Which choices did we approve? | [D1–D17 decisions](docs/DECISIONS.md) |
| What changed, and why? | [Chronological run history and clarifications](docs/RUN-HISTORY.md) |
| Which code produced the experiments? | [Pinned reproduction instructions](docs/REPRODUCTION.md) |
| How were target labels and the domain probe handled? | [Final evaluation protocol](docs/FINAL-EVALUATION-PROTOCOL.md) |
| What is verified and still missing? | [Repository audit and handoff](docs/REPOSITORY-AUDIT.md) |

## Official run set

The following values are transcribed from the student's Colab completion messages.
They are a navigation summary; the pending export will supply the authoritative files.
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
[tools/export_task2_evidence.py](../tools/export_task2_evidence.py) copies small evidence
without modifying runs or exporting checkpoints. Upload its ZIP for review before importing
it into `task2/results/` and `task2/provenance/`.

## Reproduction and attribution

Use [REPRODUCTION.md](docs/REPRODUCTION.md). Resume requires the exact recorded snapshot
and runtime; do not use this reorganized checkout to resume an old run.
Historical [manifests](provenance/README.md) retain their original bytes and commit context.

Implementation uses PyTorch, torchvision, NumPy, Pillow, scikit-learn, and Matplotlib.
The model uses torchvision's ImageNet-V1 ResNet-18 weights. No public DAN/DANN/CDAN
implementation was copied. Codex assisted with implementation and technical documentation;
the student must independently write the report's prose and interpretation.
