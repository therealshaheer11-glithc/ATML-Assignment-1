# Task 2 v4 adversarial-normalization pilot

This package implements a contained stability pilot for DANN and, only if separately
approved after the DANN source-only diagnostic, CDAN. It deliberately contains no
target-label evaluation command.

V4 retains every v3 setting. Its only training change is to L2-normalize each
512-dimensional feature at the DANN/CDAN discriminator input. Classification still uses
the original feature. CDAN uses normalized features with probabilities from the original
logits and does not detach either term. The rationale, authorization, source-only
evidence, and adoption rule are recorded in `docs/DECISIONS.md` and
`docs/PROTOCOL-REVISION-20260923-V4.md`.

Existing v3 Source-only and DAN runs stay fixed. V3 DANN also remains untouched while the
v4 DANN pilot is written to a new output root. The student will choose between v3 and v4
using source information only. If v4 is rejected, CDAN must be run with the unchanged v3
package. If v4 is adopted, CDAN must use this same v4 adversarial-input rule.

Read these files before execution:

1. `docs/ASSIGNMENT-PROTOCOL.md` - requirements taken from `ATML-PA1.pdf`.
2. `docs/DECISIONS.md` - D1-D17 approved choices and revisions.
3. `docs/PROTOCOL-REVISION-20260923-V4.md` - exact pilot rule and source-only rationale.
4. `task2/preregistration/DAN_STRENGTH_EXPECTATION.txt` - the student's locked
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
7. Run only `dann` in the new v4 pilot output root.
8. Compare v3 and v4 DANN using source information only and record the student's choice.
9. Run `cdan` under the approved version. Do not evaluate target labels yet.

The v4 DANN run creates `experiment_lock.json`. Every later v4 run must match its code,
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
