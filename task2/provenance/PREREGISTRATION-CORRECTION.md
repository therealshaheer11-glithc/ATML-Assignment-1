# Technical correction to the recorded study expectations

Added: **22 September 2026, after final target evaluation and verification**.

Original record: [dan_strength_preregistration.json](dan_strength_preregistration.json), recorded at `2026-09-22T14:21:33.152051+00:00`. That file is preserved byte-for-byte.

Two descriptions in the original expectations do not match the Task 2 protocol:

1. The domain-separability diagnostic distinguishes **pooled source examples from Sketch target examples**. It has two balanced groups, so chance accuracy is **50%**. The original reference to three source domains and approximately 33% chance was incorrect.
2. Task 2 DAN aligns the feature distribution of the **pooled labeled source batch with the unlabeled Sketch batch**. The original source-to-source description was incorrect. The implemented loss follows the source-to-target protocol.

The implemented logistic probe and DAN loss already use these required definitions. No experimental settings, checkpoints, predictions, metric values, or original expectations were altered by this correction. This post-evaluation clarification is not a replacement preregistration or a revised pre-experiment prediction.

This file is technical repository documentation. The student must independently write the assignment PDF's language, interpretation, and analysis.
