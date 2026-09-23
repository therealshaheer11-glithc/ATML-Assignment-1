# Task 2 final evaluation protocol - v5

This phase is permitted only after the six official checkpoints have been frozen and
independently verified using source information. It is the first corrected-run phase
that reads Sketch class labels. No output from this phase may change a checkpoint,
method version, hyperparameter, split, or selection decision.

## Frozen methods

- Source-only, DAN lambda 0.1, DAN lambda 1, and DAN lambda 10 use their verified v3
  checkpoints.
- DANN and CDAN use their verified v4 checkpoints after the documented source-only v4
  adoption decision.
- The main assignment comparison uses Source-only, DAN lambda 1, DANN, and CDAN.
- The controlled study reports Source-only and all three requested DAN strengths.

## Classification evaluation

Use the frozen evaluation transform and all 3,929 Sketch images. Parse the class name
from the Sketch parent directory only inside `task2/evaluate_frozen.py`, after the
freeze hash, source-audit hash, checkpoint hashes, dataset snapshot, split, and runtime
have passed validation. Report target accuracy and seven-label macro-F1
(`zero_division=0`), together with target-accuracy change relative to Source-only.

Report source accuracy and macro-F1 for Photo, Art Painting, and Cartoon from the
independently recomputed source audit, plus their unweighted means. Save every target
prediction, per-class target accuracy, change from Source-only, all confusion matrices,
dominant confusions, and high-confidence failure-example paths.
For each true class, retain the three highest-confidence errors in canonical dataset
order after sorting by confidence. This presentation-only selection does not affect any
metric. Report every class tied for a largest change or dominant confusion rather than
silently breaking such ties.

## Domain-separability probe

Use the raw 512-dimensional frozen backbone features. Pool all 1,213 source-validation
features in Photo, Art Painting, Cartoon order. With NumPy seed 6304, sample 1,213 of
the 3,929 target features without replacement. Reuse this exact balanced example set
and one stratified 70/30 partition for every method. Sort the sampled target dataset
indices after sampling only to preserve canonical output order; sorting does not change
which examples were selected.

Fit `StandardScaler` only on the probe-training portion. Then fit balanced binary
`LogisticRegression(C=1, solver="lbfgs", max_iter=2000, random_state=6304)` and report
ordinary held-out accuracy. Source is domain label 0 and Sketch is domain label 1;
chance is 50%. Save the selected target IDs, split indices, predictions, probabilities,
and confusion matrix.

## Evidence produced

- Required main four-method comparison table.
- Six-run and DAN-strength tables.
- Domain separability versus target recognition plot.
- Per-class changes and selected failure examples.
- Confusion matrices and heatmaps.
- Complete training curves from the frozen histories.
- Machine-readable predictions, metrics, protocol identities, and file hashes.

The evaluator writes through a partial directory and publishes the final directory only
after every run and artifact succeeds. It refuses to overwrite a completed evaluation.
