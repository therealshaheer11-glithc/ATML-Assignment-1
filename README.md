# ATML Programming Assignment 1

Code, experiment protocols, results, and reproducibility evidence for all four tasks.

| Task | Guide | Current status |
| --- | --- | --- |
| 1 — STL-10 inductive biases | [Task 1](task1/README.md) | Code, split manifests, predictions, metrics, and figures published |
| 2 — PACS domain adaptation | [Task 2](task2/README.md) | Official histories, frozen results, plots, and provenance published |
| 3 — PACS domain generalization | [Task 3](task3/README.md) | Training, diagnostics, final experiment lock, and one-time Sketch evaluation completed |
| 4 — Open-set recognition | [Task 4](task4/README.md) | Training, target-free evaluation lock, and one-time CIFAR-100 evaluation completed |

## Repository layout

```text
task1/       STL-10 implementation, configurations, results, and reproduction guide
task2/       PACS domain-adaptation code, decisions, results, and provenance
task3/       PACS domain-generalization code, locked protocols, results, and provenance
task4/       CIFAR open-set code, locked configurations, tests, results, and provenance
shared/      Shared PACS data loading, fixed split, and MMD implementation
tests/       Repository-level locked-choice and saved-evidence checks
tools/       Saved-evidence verification and Colab export utilities
```

The [Task 1 environment](colab-environment.json) and
[Colab add-on dependencies](requirements-colab.txt) apply to Task 1. Later tasks use
the environments recorded in their respective run and provenance files.

Datasets, model checkpoints, and large feature caches remain outside Git. Each task
guide documents its storage locations and the requirements for reproduction from a
fresh checkout.

## Repository verification

Install the lightweight testing dependency and run all repository checks from the
repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests task3/tests task4/tests
```

The Task 3 tests can also be run without pytest:

```bash
python -m unittest discover -s task3/tests -v
```

## Provenance and experiment status

- The first Task 2 attempt is preserved on
  [archive/task2-attempt-1-20260923](https://github.com/therealshaheer11-glithc/ATML-Assignment-1/tree/archive/task2-attempt-1-20260923).
- The official Task 2 comparison combines **V3 Source-only/DAN** and
  **V4 DANN/CDAN**. The [run history](task2/docs/RUN-HISTORY.md) documents the
  corrections, stability pilots, adoption decision, and checkpoint freeze.
- Task 2 reproduction must use the
  [pinned reproduction instructions](task2/docs/REPRODUCTION.md), because the
  organized documentation checkout has a different code-tree identity from the
  historical training snapshots.
- The [Task 2 repository audit](task2/docs/REPOSITORY-AUDIT.md) records the completed
  checks.
- Task 3’s final lock, one-time Sketch evaluation, and artifact identities are recorded
  in its [run log](task3/provenance/RUN_LOG.md).
- Task 4’s selected checkpoints, frozen evaluation protocol, final tables, failure
  analysis, and artifact identities are indexed in the
  [Task 4 results guide](task4/results/README.md).

## Attribution

ChatGPT/Codex was used as a coding assistant for implementation support, debugging,
testing, verification, and repository documentation. The student reviewed and is
responsible for every submitted line of code and for the execution of all experiments.

Materially reused external code, pretrained models, research papers, and reference
implementations are identified in the relevant task guides and protocol documents.
The PDF report is written independently by the student without generative AI, as
required by the assignment.
