# Task 2 corrected rerun v3 - normalized-MMD training phase

This package implements the locked source-only, DAN, DANN, and CDAN training protocol
for the corrected Task 2 rerun. It deliberately contains no target-label evaluation
command. Target evaluation is a separate phase after all six checkpoints are complete
and frozen.

The v3 protocol retains global L2 gradient-norm clipping at 20 for every method. For DAN,
it additionally L2-normalizes each 512-dimensional feature only inside the MMD loss;
classification still uses the original feature. The unclipped v1 and clipped v2 runs are
diagnostic pilots only and must not be included in the official comparison. The
TA-authorized rationale and source-only evidence are recorded in `docs/DECISIONS.md`,
`docs/PROTOCOL-REVISION-20260923.md`, and
`docs/PROTOCOL-REVISION-20260923-V3.md`.

Read these files before execution:

1. `docs/ASSIGNMENT-PROTOCOL.md` - requirements taken from `ATML-PA1.pdf`.
2. `docs/DECISIONS.md` - D1-D16 approved choices and revisions.
3. `task2/preregistration/DAN_STRENGTH_EXPECTATION.txt` - the student's locked
   expectation and disclosure, written before the official clipped runs. The training
   command refuses to run if a `PENDING` placeholder is present.

## What this phase can access

- Source train records contain paths and class IDs.
- Source validation records contain paths and class IDs.
- Target adaptation records contain only paths and opaque IDs. The target dataset
  class is label-blind and returns no class label.
- Checkpoint selection uses only the unweighted mean macro-F1 over the three source
  validation domains.

## Order of operations in a fresh Colab runtime

1. Record and approve the runtime preflight.
2. Upload and extract this exact code archive.
3. Download and verify the PACS archive, then extract it.
4. Verify the locked preregistration file and its recorded hash.
5. Run `task2.preflight`; this checks data, split, configurations, and environment but
   does not train and does not access target labels.
6. Create the one common ImageNet-V1 plus seven-class-head initialization.
7. Train the six runs sequentially in the same runtime/output root:
   `source_only`, `dan_0p1`, `dan_1`, `dan_10`, `dann`, `cdan`.
8. Freeze and audit the six selected checkpoints before adding target-label evaluation.

The first run creates `experiment_lock.json`. Every later run must match its code,
split, initialization, preregistration, dataset snapshot, runtime, GPU, and worker
count. A completed run stores `best.pt`, `last.pt`, `history.csv`,
`best_source_validation.json`, and `run.json`.

## Resume rule

Use `--resume` only in the same run directory and only after an epoch-complete
`last.pt` exists. The command rejects any identity mismatch. An interruption during an
epoch replays that epoch from the last complete checkpoint.

## Authorship note

This implementation was developed with OpenAI Codex as coding assistance. It uses
PyTorch, torchvision, NumPy, Pillow, and scikit-learn APIs; it does not copy a public
DAN/DANN/CDAN implementation. The student must inspect and understand every submitted
line. Codex output must not be used as report prose, interpretation, or analysis.
