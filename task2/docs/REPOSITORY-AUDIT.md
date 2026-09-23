# Repository audit and handoff — 23 September 2026

## Scope

Reviewed GitHub `main` at `edecd5b9429799cc51c9e96625191beaf45562af` and the preserved
archive branch. This is a source-code and saved-evidence audit. It does not claim fresh
GPU reproduction or inspection of checkpoints that are only in the student's Drive.

## Verified locally

- Task 1: disjoint 4,000/1,000 train/validation indices and class counts; 500 test indices,
  50 per class; all three head checkpoint selections reconstructed from histories.
- Task 1: clean accuracy/F1/confidence, color, patch and 48 directional translation results
  recomputed from published predictions; 200 cue scores and frozen selection hashes
  checked; cosine means/medians and three UMAP coordinate/figure hashes verified.
- Task 2: V4's 27 manifested files and V5's three manifested files match their pinned
  commits; fixed split digest matches the recorded source split.
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

They explicitly report Task 2's final evidence as pending. See
[run-history clarifications](RUN-HISTORY.md) for the precise meanings of gradient logs,
CDAN norm logs, and the presentation-only failure-example tie-order limitation.

## Cleanup

The root README is now a short index. The full Task 1 reproduction guide has moved to
`task1/README.md` with repaired links. Task 2 has a current guide, chronological account,
pinned reproduction commands, and clear separation between historical packaging status
and subsequent reported execution. No branch or experiment history is deleted or rewritten.
Original manifests remain intact and are interpreted against their pinned commits.

## Remaining before Task 2 publication is complete

1. Run `tools/export_task2_evidence.py` in Colab. It reads existing outputs, checks the
   freeze/source-audit hashes, six checkpoint hashes and source selections, and recomputes
   target accuracy/F1 from saved predictions. It copies small evidence only.
2. Download the **latest corrected notebook** as `.ipynb` from Colab. The old first-attempt
   notebook does not cover the corrected recovery, adoption, and freeze cells. Supply it
   via the exporter's `--notebook` option or send it separately for inspection.
3. Review the export, including any missing optional pilot evidence. Import its final
   results, training histories and provenance without changing original bytes. Check
   complete per-class/confusion/probe tables and figures against the assignment.
4. Update the Task 2/root status to “published” only after those files reach GitHub and
   their paths/hashes are checked. Add exact orchestration instructions from the notebook.
5. The student writes the report using the evidence map and documents TA permissions,
   source-based revisions, prior target exposure, and unsuccessful runs. Verify its
   figures, required analyses, format and page limit separately.

Until steps 1–4 are complete, Task 1's published evidence is ready for report use; Task 2
has completed-run console evidence but is not yet a fully published, self-contained
experiment record. No new training or target-driven tuning is needed for this handoff.
