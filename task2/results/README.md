# Task 2 saved results

These are small, immutable records from the official frozen-checkpoint experiment.
PACS images and neural checkpoints are excluded from Git.

## Official training

[`training/`](training/) contains `history.csv`, `run.json`, and
`best_source_validation.json` for each official run:

- V3: Source-only, DAN λ=0.1, DAN λ=1, and DAN λ=10.
- V4: DANN and CDAN, following the source-only adoption decision.

Each run record identifies its selected epoch and checkpoint SHA-256. Checkpoint files
remain in Drive.

## Frozen final evaluation

[`final/`](final/) contains:

- `main_four_comparison.csv`, `all_six_comparison.csv`, and
  `dan_strength_study.csv`;
- all 3,929 Sketch predictions for each run, per-class results, class changes,
  confusion matrices, dominant confusions, and selected failure examples;
- the shared balanced domain-probe partition and held-out predictions;
- complete training curves and aggregate/class-level plots;
- `final_results.json`, completion metadata, and
  `EVALUATION-MANIFEST.json` with hashes for all 25 evaluation artifacts.

The original manifested scatter is retained. [`presentation/`](presentation/) contains
a presentation-only copy with an external legend because the original point labels
overlap. Its JSON record identifies the unchanged source CSV and output hashes.

The six checkpoints were frozen before target labels were exposed. Source validation
selected every checkpoint; Sketch labels were used only in this final evaluation.

## Verification

From the repository root:

```bash
python tools/verify_saved_evidence.py
```

This recomputes saved classification accuracy and macro-F1 from the prediction CSV,
checks source-selection histories and the balanced domain probe, and verifies all
published hashes. It performs no neural inference.

The run chronology, failed pilots, protocol revisions, and interpretation limits are in
[`RUN-HISTORY.md`](../docs/RUN-HISTORY.md). Original orchestration records and the
corrected notebook are under [`provenance/`](../provenance/).
