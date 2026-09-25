# Task 3 final-lock and Sketch-evaluation protocol

## Purpose and timing

This protocol was approved only after all eight checkpoints had finished training and
all source-only diagnostics had been completed. It does not revise, replace, or tune
any checkpoint. Block 15 creates the immutable experiment lock without reading target
records, target labels, archive members, Task 2 target results, or Sketch images.
Block 16 is the only Task 3 path permitted to access those items, and it runs only
after the exact Block 15 lock hash is supplied as explicit authorization.

## Frozen checkpoints and reporting groups

The lock contains exactly these checkpoints, in this fixed evaluation order:

1. ERM (the exact Task 2 source-only checkpoint);
2. original DAN-DG, lambda 1;
3. standard non-adaptive SAM, rho 0.05;
4. original DAN-DG, lambda 0.1;
5. original DAN-DG, lambda 10;
6. supplementary bandwidth-floor DAN-DG, lambda 0.1;
7. supplementary bandwidth-floor DAN-DG, lambda 1; and
8. supplementary bandwidth-floor DAN-DG, lambda 10.

The assignment's official main comparison remains ERM versus original DAN-DG lambda
1 versus SAM. The original lambda set is reported as the prescribed controlled study.
The bandwidth-floor set is reported separately as a supplementary, explicitly
post-diagnostic research variant and never substitutes for the original protocol.

## One-time target evaluation

After every pre-target gate passes, Block 16 extracts exactly the 3,929 locked Sketch
records from the already authenticated PACS archive. Every checkpoint is evaluated on
all 3,929 examples in identical protocol order, with resize to 256 by 256, center crop
to 224 by 224, ImageNet normalization, batch size 64, no shuffle, and no augmentation.
The fixed metrics are target accuracy, macro-F1 over all seven classes, per-class
accuracy and support, a seven-by-seven confusion matrix, and accuracy change relative
to ERM. No result can trigger retraining, checkpoint replacement, or a second target
evaluation.

## Per-class changes and examples

For each non-ERM checkpoint, per-class accuracy is compared with ERM. The class with
the largest improvement and the class with the largest degradation are selected; an
exact tie is resolved by the fixed PACS class order. Within each selected class, up to
three disagreement examples are selected by lexicographically sorted image path.
The saved row includes the Task 3 ERM and candidate predictions and the corresponding
Task 2 source-only and target-aware DAN lambda-1 predictions. This is analysis of the
single frozen evaluation, not a basis for model selection.

## Task 2 comparison

Block 16 authenticates the existing Task 2 final results and prediction table before
reading them. It reports the assignment-requested comparison between Task 2
target-aware DAN lambda 1 and Task 3 target-free DAN-DG lambda 1, including aggregate
target metrics and the corresponding predictions for selected examples.

## Audit outputs

Block 15 saves `final_experiment_lock.json` and a lock-completion record. Block 16
saves the authorization record, full metrics, predictions, selected examples, and a
completion manifest containing SHA-256 identities for every output. Existing final
outputs are never overwritten.
