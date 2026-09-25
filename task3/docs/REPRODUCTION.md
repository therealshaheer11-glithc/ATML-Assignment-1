# Task 3 Reproduction and Verification

Task 3 was executed across several reviewed repository snapshots. Use the pinned commits below when reproducing a historical stage. The notebook blocks under `task3/notebook_blocks/` are preserved execution records and should not be run against the moving tip of `main`.

## Published-result verification

The final machine-readable results, predictions, locks, histories, and execution records are already published. Verification of this evidence does not require PACS images, model checkpoints, or access to Sketch.

From a current repository checkout, run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests task3/tests task4/tests
python tools/verify_saved_evidence.py
```

The Task 3 tests can also be run without pytest:

```bash
python -m unittest discover -s task3/tests -v
```

The final result files are indexed in [`../results/README.md`](../results/README.md), and the complete execution history is recorded in [`../provenance/RUN_LOG.md`](../provenance/RUN_LOG.md).

## Historical execution snapshots

The following commits identify the exact repository stages used during the experiment:

| Stage | Repository commit |
|---|---|
| Source-only data preparation | `12c9c772579d8fe8d129f6345f37043064c9c60c` |
| Prescribed DAN-DG and SAM implementation and training | `19208b4c62acb980fb3246f30e062784b90d8dfc` |
| Supplementary bandwidth-floor calibration and training | `dc3acfdf547e7bc29bd381b3fe05e271879f18d0` |
| Complete source-only diagnostics | `136e28b556d38c99e486e2d53ddff33d921fbfc6` |
| Final experiment lock and one-time Sketch evaluation | `31618caebaf42acb18dd657407d22f03b4a2464f` |

To inspect one historical snapshot without changing the repository history:

```bash
git switch --detach <commit>
```

After inspection, return to the current repository with:

```bash
git switch main
```

Do not substitute the latest `main` commit when authenticating a historical execution record. Documentation and tests added after execution legitimately change the current file tree.

## Runtime and required external artifacts

The recorded Task 3 runtime was:

- Python 3.13.15
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- NumPy 2.1.3
- scikit-learn 1.6.1
- CUDA 12.8
- cuDNN 91900
- Tesla T4 GPU

The experiment also depends on external artifacts intentionally excluded from Git:

- the PACS archive;
- the Task 2 common initialization;
- the selected Task 2 ERM checkpoint;
- selected Task 3 checkpoints; and
- large intermediate feature artifacts.

Their expected locations and SHA-256 identities are recorded in [`../provenance/RUN_LOG.md`](../provenance/RUN_LOG.md) and [`../provenance/final_experiment_lock.json`](../provenance/final_experiment_lock.json).

A fresh user without those checkpoint files may inspect and verify the published evidence, but exact model inference requires either the authenticated external artifacts or reproduction of the corresponding target-free training stages.

## Reproducing prescribed source-only training

Use commit:

```text
19208b4c62acb980fb3246f30e062784b90d8dfc
```

At that snapshot, the implementation-verification stage ran 14 target-free tests and authenticated executable code-tree SHA-256:

```text
4ee16e4b2b66fa051e6571a666a935e6721681e9ac8c1325d5a494ffda528e44
```

The approved prescribed runs are:

- `dan_dg_1`, the main DAN-DG comparison;
- `dan_dg_0p1`, the lower-strength controlled run;
- `dan_dg_10`, the higher-strength controlled run; and
- `sam`, the main SAM comparison.

Only Photo, Art Painting, and Cartoon may be available during training, selection, or source-side diagnostics. Sketch must remain inaccessible.

The original epoch-level histories are published under [`../results/training/prescribed/`](../results/training/prescribed/).

## Supplementary bandwidth-floor study

Use commit:

```text
dc3acfdf547e7bc29bd381b3fe05e271879f18d0
```

Its executable code-tree SHA-256 was:

```text
34d2778e76a5a7522a9c2eac2682eb4d2a09ee9de0e88b6d04d7d0a68e26ce13
```

This separately preregistered study produced `dan_dg_floor_0p1`, `dan_dg_floor_1`, and `dan_dg_floor_10`. It is supplementary and must not replace the assignment-prescribed adaptive-bandwidth runs.

The histories are published under [`../results/training/supplementary/`](../results/training/supplementary/).

## Final evaluation boundary

The final lock and one-time Sketch evaluation were completed at commit:

```text
31618caebaf42acb18dd657407d22f03b4a2464f
```

The final experiment lock was created before any Sketch access. The authorized evaluation then processed all 3,929 locked Sketch images once. No checkpoint or selection decision changed afterward.

This one-time evaluation is a historical completed action and should not be rerun merely to verify the repository. Use the saved lock, completion records, predictions, and machine-readable results for verification.
