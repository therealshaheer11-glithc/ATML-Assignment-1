# ATML Programming Assignment 1

Code, experiment protocols, and saved evidence. Start with the task guides below.

| Task | Guide | Current status |
| --- | --- | --- |
| 1 — STL-10 inductive biases | [Task 1](task1/README.md) | Code, split manifests, predictions, metrics, and figures published |
| 2 — PACS domain adaptation | [Task 2](task2/README.md) | Code, official histories, frozen results, plots, and provenance published |
| 3 — PACS domain generalization | [Task 3](task3/README.md) | Training, diagnostics, final experiment lock, and one-time Sketch evaluation completed |
| 4 — Open-set recognition | [Task 4](task4/README.md) | Training, target-free lock, and one-time CIFAR-100 evaluation completed |

## Repository layout

```text
task1/       STL-10 code, configuration, reproduction guide, and results
task2/       PACS UDA code, configurations, decision history, and provenance
task3/       PACS DG code, frozen protocols, execution log, and evaluation tooling
task4/       CIFAR open-set code, locked configurations, tests, and execution guide
shared/      PACS data loading, fixed split, and MMD implementation
tests/       Checks for locked Task 2 choices and final evaluation
tools/       Saved-evidence verification and Colab evidence export
```

[Task 1 environment](colab-environment.json) and [Colab add-on dependencies](requirements-colab.txt)
apply to Task 1. Task 2 uses the environment recorded in its own run records.
Datasets, checkpoints, and feature caches remain outside Git; each task guide identifies
their storage locations and the limits of reproducing from a fresh clone.

## Repository verification

Install the lightweight test dependency, then run the repository checks from the root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests task3/tests
```

The Task 3 suite can also be run without pytest:

```bash
python -m unittest discover -s task3/tests -v
```

## Provenance and experiment status

- The first Task 2 attempt is preserved on
  [archive/task2-attempt-1-20260923](https://github.com/therealshaheer11-glithc/ATML-Assignment-1/tree/archive/task2-attempt-1-20260923).
- The official Task 2 comparison combines **V3 Source-only/DAN** and **V4 DANN/CDAN**.
  [Run history](task2/docs/RUN-HISTORY.md) explains the corrections, stability pilots,
  adoption decision, and checkpoint freeze.
- Use the [pinned reproduction instructions](task2/docs/REPRODUCTION.md) for Task 2;
  the latest documentation checkout has a different code-tree identity from the training snapshots.
- [Repository audit](task2/docs/REPOSITORY-AUDIT.md) lists completed checks and the remaining
  evidence needed before declaring Task 2 publication complete.

## Authorship

ChatGPT/Codex assisted with implementation, debugging, verification, and technical
documentation. External code and pretrained models are attributed in each task guide.
The student must understand the submitted code and write the report's prose,
interpretation, and analysis independently, as required by the assignment.
