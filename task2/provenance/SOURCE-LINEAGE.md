# Task 2 source and run lineage

This is the complete map from every material Task 2 stage to its exact source and saved
evidence. Diagnostic and failed runs remain visible but are excluded from the official
comparison. Large model checkpoints and PACS images remain in Drive or external storage.

| Stage | Exact source | Saved evidence | Status |
| --- | --- | --- | --- |
| Original attempt | branch `archive/task2-attempt-1-20260923`, head `e7b72cd0ae05aeca301b877c041c3c3730335ee0` | Code, configurations, histories, first final analysis, verification, and excluded interrupted DAN λ=0.1 record on that branch | Archived and excluded after the protocol reset |
| Exploratory Kaggle λ=10 | No repository snapshot; settings and runtime differed from the official experiment | Narrative only in `docs/RUN-HISTORY.md`; the attempted matched rerun stopped at its environment gate before training | Excluded; cannot establish a causal explanation |
| Corrected V1 | [`source-packages/ATML-PA1-Task2-corrected-code-v1.zip`](source-packages/ATML-PA1-Task2-corrected-code-v1.zip), SHA-256 `f85cee7f58ccad18d9820bd11b685c1463a29ad0130042c35adc3d73c98a0a6e` | `pilots/v1/` plus `versions/v1/`; Source-only, DAN λ=0.1, and failed DAN λ=1 | Diagnostic; excluded from official tables |
| Clipped V2 | commit `6e0aa49d7a678091840527a5d5a2f304296860ba` | `pilots/v2/dan_1/` and `versions/v2/` | Diagnostic; clipping retained |
| Normalized-MMD V3 | commit `b96f184184b39a3c9c47ee55d3294311a0e737e8` | Official Source-only and DAN histories under `../results/training/`; failed DANN pilot under `pilots/v3/dann/`; V3 lock/preflight under `versions/v3/` | Official Source-only and DAN λ∈{0.1,1,10}; V3 DANN superseded |
| Adversarial-normalized V4 | commit `d26997b22d3b7722e2ecc828dde1445244afc04b` | Official DANN/CDAN histories under `../results/training/`; adoption, diagnostic, initialization check, lock, and preflight under `versions/v4/` | Official DANN and CDAN |
| Six-run freeze | V3 and V4 snapshots above | `freeze/source_checkpoint_audit.json`, `freeze/task2_checkpoint_freeze.json`, and `freeze/evaluation_code_lock.json` | Frozen before corrected-run target-label evaluation |
| Final evaluator V5 | commit `edecd5b9429799cc51c9e96625191beaf45562af` | 25 manifested files under `../results/final/`, including all 3,929 target predictions and probe predictions | Official final analysis; no retraining |
| Evidence publication | `main` publication commits `458499f8bb0270e93318d952d945a5c2938a7eed` and `c3b9599ecb0845733fd5a082ac866e9a66dedb01` | `FINAL-EVIDENCE-EXPORT.json`, corrected notebook, results, plots, and presentation-only readable separability plot | Repository handoff |

## V1 clarification

Corrected V1 ran from a ZIP uploaded directly to Colab, so no Git commit captured that
exact package before V2 replaced it. The ZIP's internal manifest covers 25 source files.
Its packaged expectation file is the pre-lock placeholder. The completed expectation
actually used by the V1 runs is independently retained at
`versions/v1/preregistration/DAN_STRENGTH_EXPECTATION.txt`, and its SHA-256 is recorded in
the V1 run records.

## External artifacts intentionally absent from Git

- PACS images and archive; archive SHA-256 and source URL are in `docs/REPRODUCTION.md`.
- Common initialization and model checkpoints; identities and original Drive paths are
  in the run records, source audit, and freeze.
- The unmatched exploratory Kaggle implementation and output; they were not part of the
  official pipeline and were explicitly discarded rather than represented as evidence.

The final report must use the frozen V3/V4 checkpoints and V5 results. Historical pilots
explain protocol changes; they must not be substituted into the official comparison.
