# ATML Programming Assignment 1

Code, experiment protocols, results, and reproducibility evidence for all four tasks.

| Task | Guide | Current status |
|---|---|---|
| 1 — STL-10 inductive biases | [Task 1](task1/README.md) | Code, split manifests, predictions, metrics, and figures published |
| 2 — PACS domain adaptation | [Task 2](task2/README.md) | Official histories, frozen results, plots, and provenance published |
| 3 — PACS domain generalization | [Task 3](task3/README.md) | Training, diagnostics, final experiment lock, and one-time Sketch evaluation completed |
| 4 — Open-set recognition | [Task 4](task4/README.md) | Training, target-free evaluation lock, and one-time CIFAR-100 evaluation completed |
| Report | [Report evidence](report/README.md) | Report figures, derived-number checks, and the Task 4 per-class analysis |

## Repository layout

```text
task1/       STL-10 implementation, configurations, results, and reproduction guide
task2/       PACS domain-adaptation code, decisions, results, and provenance
task3/       PACS domain-generalization code, locked protocols, results, and provenance
task4/       CIFAR open-set code, locked configurations, tests, results, and provenance
shared/      Shared PACS data loading, fixed split, and MMD implementation
report/      Report figure scripts, figures, and derived-number checks
tests/       Repository-level locked-choice and saved-evidence checks
tools/       Saved-evidence verification and Colab export utilities
```

Datasets, large model checkpoints, environments, and large feature or score caches
remain outside Git. Small machine-readable results, split manifests, training
histories, predictions, figures, locks, and provenance records are published in their
corresponding task directories.

## Environment scope

The repository contains several related environment records rather than one universal
environment file because the four tasks were executed at different stages.

- The [Task 1 environment](colab-environment.json) and
  [Task 1 Colab add-on dependencies](requirements-colab.txt) apply to Task 1.
- Task 2’s exact environments are recorded in its run and provenance files, with
  reproduction instructions in
  [`task2/docs/REPRODUCTION.md`](task2/docs/REPRODUCTION.md).
- Task 3’s exact runtime and historical repository snapshots are documented in
  [`task3/docs/REPRODUCTION.md`](task3/docs/REPRODUCTION.md).
- Task 4’s general dependencies are listed in
  [`task4/requirements.txt`](task4/requirements.txt), while its exact completed runtime
  is recorded in
  [`task4/provenance/runtime_preflight.json`](task4/provenance/runtime_preflight.json).

Exact experiment reproduction should use the task-specific recorded environment and
pinned repository snapshot. Installing only pytest is sufficient only when the
scientific and machine-learning dependencies are already available.

## Repository verification

From a fresh Python environment, install the general scientific dependencies used by
the repository checks and the lightweight test dependency:

```bash
python -m pip install -r task4/requirements.txt
python -m pip install -r requirements-dev.txt
```

Then run all repository tests from the repository root:

```bash
python -m pytest -q tests task3/tests task4/tests
```

The Task 3 suite can also be run without pytest:

```bash
python -m unittest discover -s task3/tests -v
```

To verify the published Task 1 and Task 2 evidence, historical package manifests,
saved metrics, predictions, plots, and local documentation links, run:

```bash
python tools/verify_saved_evidence.py
```

The saved-evidence verifier expects a normal Git clone containing the repository
history because it authenticates files against pinned historical commits. These
verification commands inspect published evidence; they do not rerun model training or
the completed one-time target evaluations.

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
- The [Task 2 repository audit](task2/docs/REPOSITORY-AUDIT.md) records its completed
  checks.
- Task 3’s pinned snapshots and verification instructions are documented in its
  [reproduction guide](task3/docs/REPRODUCTION.md). Its published final results and
  training histories are indexed in the
  [Task 3 results guide](task3/results/README.md).
- Task 4’s selected-checkpoint records, frozen evaluation protocol, final tables,
  failure analysis, and artifact identities are indexed in the
  [Task 4 results guide](task4/results/README.md). The historical scope of its
  publication manifest is documented in the
  [Task 4 provenance guide](task4/provenance/README.md).

## Attribution

ChatGPT/Codex was used as a coding assistant for implementation support, debugging,
testing, verification, and repository documentation. The student reviewed and is
responsible for every submitted line of code and for the execution of all experiments.

Materially reused external code, pretrained models, research papers, and reference
implementations are identified in the relevant task guides and protocol documents.

