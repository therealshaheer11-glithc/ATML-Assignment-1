# Task 3 Results

This directory contains the published evidence for the Task 3 PACS domain-generalization experiment.

## Experiment status

The prescribed source-only training, source-side diagnostics, final experiment lock, and one-time evaluation on the locked Sketch target split are complete.

No Sketch image or label was used for training, checkpoint selection, hyperparameter selection, or source-side diagnostics. The final Sketch evaluation was performed only after all checkpoint and evaluation choices had been frozen.

## Final evaluation evidence

The [`final/`](final/) directory contains:

- [`final_results.json`](final/final_results.json): machine-readable source-validation and final Sketch results for all evaluated models.
- [`target_predictions.csv`](final/target_predictions.csv): predictions for all 3,929 images in the locked Sketch split.
- [`selected_change_examples.csv`](final/selected_change_examples.csv): selected prediction changes and failure cases for analysis.

The complete experiment lock, completion records, artifact identities, and execution history are stored under [`../provenance/`](../provenance/).

## Prescribed training histories

The [`training/prescribed/`](training/prescribed/) directory contains the original epoch-level histories for:

- `dan_dg_0p1`: prescribed DAN-DG controlled-study run with lambda 0.1.
- `dan_dg_1`: prescribed DAN-DG main-comparison run with lambda 1.
- `dan_dg_10`: prescribed DAN-DG controlled-study run with lambda 10.
- `sam`: prescribed SAM main-comparison run with rho 0.05.

Each `history.csv` records classification loss, the MMD penalty where applicable, gradient diagnostics, and validation accuracy and macro-F1 for Photo, Art Painting, and Cartoon.

The ERM baseline was reused unchanged from Task 2. Its training history is published at [`../../task2/results/training/source_only/history.csv`](../../task2/results/training/source_only/history.csv).

## Supplementary research histories

The [`training/supplementary/`](training/supplementary/) directory contains the epoch-level histories for the separately preregistered initialization-anchored bandwidth-floor study:

- `dan_dg_floor_0p1`
- `dan_dg_floor_1`
- `dan_dg_floor_10`

These runs are supplementary research variants. They do not replace the PDF-prescribed adaptive-bandwidth DAN-DG runs, and they did not change the locked main comparison or any post-evaluation decision.

## Storage boundary

Large model checkpoint files remain in Google Drive and are intentionally excluded from Git. Their identities are recorded in the Task 3 experiment lock and completion records.
