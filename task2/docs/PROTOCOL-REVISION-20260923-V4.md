# Task 2 v4 protocol revision - 23 September 2026

## Status

This is a contained adversarial-normalization pilot approved after reviewing the v3
DANN source-only training diagnostics. V3 outputs remain unchanged. The v4 package must
use a new output root. No target labels may be accessed when deciding whether to adopt
this revision.

## Source-only evidence prompting the pilot

V3 DANN completed six epochs and selected epoch 1 using the locked source-validation
rule. Its best mean source-validation macro-F1 was 0.6049088768. The maximum recorded
pre-clipping gradient norm was 40,455,498.9988, and 93.4043% of updates were clipped.
The selected-epoch training domain accuracy was 0.5203014316 at average GRL strength
0.0826106062. The selected checkpoint SHA256 was
`321e3389a246ad179d8219584d04f9b089c8a13d7ef5512d34bceab4f0f676e4`.
Target labels were not accessed.

## Approved pilot rule

For DANN and CDAN, compute a per-example L2 normalization of the 512-dimensional feature
immediately before the domain-discriminator input:

```text
f_adv = f / ||f||_2
```

DANN sends `f_adv` to the discriminator. CDAN sends
`f_adv outer softmax(classifier_logits)` to the discriminator. The probabilities are
computed from the original classifier logits, and neither term is detached. The class
classifier continues to use the original `f`.

No epsilon or clamp is introduced. A zero or non-finite feature norm stops the run. The
code logs the mean norm before and after normalization in every epoch so the application
of the rule is auditable.

No other setting changes. In particular, retain the assignment GRL schedule, unit
domain-loss weight, discriminator architecture, global L2 gradient clipping at 20,
AdamW settings, common initialization, source split, sampling, augmentation, early
stopping, and source-only checkpoint-selection rule.

## Authorization and adoption rule

The course TA permitted documented normalization to stabilize non-convergent training.
The student explicitly approved this v4 pilot. First run DANN under v4 in a new output
root. Compare v3 and v4 using source-validation macro-F1, complete source training
history, gradient norms, clipping fraction, domain accuracy, and source prediction
distribution. The student must explicitly approve adoption. If adopted, CDAN uses v4;
otherwise v4 remains diagnostic and CDAN uses v3. Target labels cannot be used in this
choice.
