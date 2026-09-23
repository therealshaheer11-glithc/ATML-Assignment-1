# Repository audit and handoff — 23 September 2026

## Scope

Reviewed GitHub `main` at `edecd5b9429799cc51c9e96625191beaf45562af`, the preserved
archive branch, and the final evidence export with SHA-256
`ac4b4db8f4e60f3dcb710be4a483eb0b763518c8a0785d7996382f33c690164e`.
This is a source-code and saved-evidence audit. It does not claim fresh GPU reproduction
or local loading of checkpoints that remain in the student's Drive.

## Verified locally

- Task 1: disjoint 4,000/1,000 train/validation indices and class counts; 500 test indices,
  50 per class; all three head checkpoint selections reconstructed from histories.
- Task 1: clean accuracy/F1/confidence, color, patch and 48 directional translation results
  recomputed from published predictions; 200 cue scores and frozen selection hashes
  checked; cosine means/medians and three UMAP coordinate/figure hashes verified.
- Task 2: V4's 27 manifested files and V5's three manifested files match their pinned
  commits; fixed split digest matches the recorded source split.
- Task 2: the 25 final evaluation artifacts match their manifest; the freeze and source
  audit hashes agree; the six official histories independently select the frozen epochs;
  3,929 unique target prediction rows reproduce every target accuracy and seven-class
  macro-F1; the export reports no missing evidence.
- All 14 existing Task 2 locked-choice/evaluation tests pass locally. These are focused
  functional checks, not proof that every notebook execution was correct.
- No tracked checkpoints, datasets, caches, `.DS_Store`, or obvious duplicate upload
  archives were found. Task 1's many small evidence files are intentional and retained.
- Training/evaluation Python, configurations, fixed split, expectations, existing metrics,
  figures, and historical manifests are unchanged by this organization pass.

Run the saved-evidence checks without GPU or model weights:

```bash
python tools/verify_saved_evidence.py
```

They verify the published Task 2 evidence without loading neural checkpoints. See
[run-history clarifications](RUN-HISTORY.md) for the precise meanings of gradient logs,
CDAN norm logs, and the presentation-only failure-example tie-order limitation.

## Cleanup

The root README is now a short index. The full Task 1 reproduction guide has moved to
`task1/README.md` with repaired links. Task 2 has a current guide, chronological account,
pinned reproduction commands, and clear separation between historical packaging status
and subsequent reported execution. No branch or experiment history is deleted or rewritten.
Original manifests remain intact and are interpreted against their pinned commits.

## Remaining submission work

The code and small evidence for Tasks 1 and 2 are ready for report use. The student must
write the report, document TA permissions, source-based revisions, prior target exposure,
and unsuccessful runs, then verify the report's figures, required analyses, format, and
page limit. Large checkpoints and datasets remain external by design. No new Task 2
training or target-driven tuning is needed.
