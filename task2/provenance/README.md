# Task 2 provenance

The package manifests remain at their original paths, preserved byte-for-byte during cleanup.
They describe the files at their pinned commits, not the current documentation tree.
Their `training_started: false` fields describe packaging time, not present run status.

| Historical manifest | Original path | Authoritative snapshot |
| --- | --- | --- |
| [V4 package](../PACKAGE-MANIFEST.json) | `task2/PACKAGE-MANIFEST.json` | `d26997b22d3b7722e2ecc828dde1445244afc04b` |
| [V5 evaluator](../FINAL-EVALUATION-MANIFEST.json) | `task2/FINAL-EVALUATION-MANIFEST.json` | `edecd5b9429799cc51c9e96625191beaf45562af` |

The original paths and all original content remain available in Git history. No training
or evaluation source, configuration, split, or existing result was changed by the cleanup.
Documentation is included in the trainer's code-tree hash, so even documentation-only
changes require using the original snapshot for exact reproduction or resumption.

Actual run records, freeze, source checkpoint audit, adoption decision, and corrected
notebook are published below this directory. Their hashes are recorded in
[`FINAL-EVIDENCE-EXPORT.json`](FINAL-EVIDENCE-EXPORT.json); do not reconstruct or edit
them from chat logs.
