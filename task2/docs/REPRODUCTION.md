# Task 2 reproduction and storage

## Preserve exact training snapshots

Training identities include documentation and package files. Never resume a saved run
from an updated `main`, mix imported modules from different extracted directories, or
silently relax identity checks. Use a fresh Python process in the matching snapshot.

| Snapshot | Exact source location | Purpose |
| --- | --- | --- |
| Original attempt | branch `archive/task2-attempt-1-20260923` at `e7b72cd0ae05aeca301b877c041c3c3730335ee0` | Excluded historical attempt |
| Corrected V1 | [`ATML-PA1-Task2-corrected-code-v1.zip`](../provenance/source-packages/ATML-PA1-Task2-corrected-code-v1.zip), SHA-256 `f85cee7f58ccad18d9820bd11b685c1463a29ad0130042c35adc3d73c98a0a6e` | Unclipped diagnostic pilots |
| Clipped V2 | commit `6e0aa49d7a678091840527a5d5a2f304296860ba` | Clipped DAN λ=1 diagnostic pilot |
| Normalized-MMD V3 | commit `b96f184184b39a3c9c47ee55d3294311a0e737e8` | Official Source-only and all three DAN strengths; superseded DANN pilot |
| Adversarial-normalized V4 | commit `d26997b22d3b7722e2ecc828dde1445244afc04b` | Official adopted DANN/CDAN |
| Final evaluator V5 | commit `edecd5b9429799cc51c9e96625191beaf45562af` | Frozen-checkpoint evaluation, no retraining |

Commits are persistent provenance; the archive branch preserves the original attempt.
V1 ran from an uploaded Colab ZIP before its source was committed, so its immutable ZIP
is retained explicitly. The ZIP contains the packaging-time expectation placeholder;
the completed locked expectation actually used by the runs is separately preserved at
`task2/provenance/versions/v1/preregistration/DAN_STRENGTH_EXPECTATION.txt`.
The [historical manifests](../provenance/README.md) verify against their
respective commits. The current README describes the completed workflow, not the old
packaging-time status.

To obtain separate snapshots from a full clone, run from the repository root:

```bash
git worktree add --detach ../atml-task2-v3 b96f184184b39a3c9c47ee55d3294311a0e737e8
git worktree add --detach ../atml-task2-v4 d26997b22d3b7722e2ecc828dde1445244afc04b
git worktree add --detach ../atml-task2-evaluation edecd5b9429799cc51c9e96625191beaf45562af
```

## Environment and data

Recorded official Colab environment: Tesla T4; Python 3.13.15; PyTorch 2.11.0+cu128;
torchvision 0.26.0+cu128; NumPy 2.1.3; scikit-learn 1.6.1; CUDA 12.8; cuDNN 91900.
These are recorded values, not a claim that current Colab defaults match them. Retain
actual run environment records, including other installed package versions. The root
`requirements-colab.txt` belongs to Task 1 and is not a Task 2 environment lock.

Verified PACS archive:
`https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE`

```text
archive SHA256:
0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102
image root used: /content/atml_pacs/pacs/images
```

Reuse `shared/splits/pacs_sketch_seed6304.json` exactly. Do not regenerate it after
seeing target results. Run `python -m task2.preflight --pacs-root PATH --output NEW_JSON`
from the selected training snapshot before a fresh reproduction.

## Training interface

The completed run used the saved common initialization and locked expectation. The
initialization generator is `python -m task2.prepare_initialization --output NEW_PT`;
exact continuation instead requires the already recorded initialization artifact.
The dataset-source argument records provenance; it does not download data.

Example command structure, with paths deliberately supplied by the reproducer:

```bash
python -m task2.train \
  --run-id dan_1 \
  --pacs-root "$PACS_ROOT" \
  --dataset-source 'PACS Dassl archive SHA256 0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102' \
  --protocol shared/splits/pacs_sketch_seed6304.json \
  --initialization "$COMMON_INITIALIZATION" \
  --preregistration task2/preregistration/DAN_STRENGTH_EXPECTATION.txt \
  --output "$NEW_RUN_DIRECTORY" --device cuda --num-workers 2
```

Run `source_only`, `dan_0p1`, `dan_1`, and `dan_10` from V3; `dann` and `cdan` from V4.
Use distinct run directories in separate version roots and the same initialization.
A reproduction produces new records, not replacements for the frozen official artifacts.
Add `--resume` only to the identical run command after an epoch-complete `last.pt` exists
and all identity/runtime checks match. Completed runs must not be retrained by accident.

## Source audit, freeze, and final evaluation

The actual source audit, freeze, V4 adoption decision, final results, official histories,
and corrected notebook are published under `task2/provenance/` and `task2/results/`.
The notebook preserves the Colab orchestration and recovery cells. Checkpoints remain in
Drive because they are large; do not substitute handwritten records for the published
files.

With the original frozen checkpoints still at the paths recorded in the freeze, the
existing evaluator can be invoked from V5 without training:

```bash
python -m task2.evaluate_frozen \
  --freeze "$FREEZE_JSON" \
  --expected-freeze-sha256 baacc896c12285216eee120b785475b69c1e3bfcee911c33a759a72c897f15b5 \
  --source-audit "$SOURCE_AUDIT_JSON" \
  --pacs-root "$PACS_ROOT" \
  --protocol shared/splits/pacs_sketch_seed6304.json \
  --output "$NEW_EVALUATION_DIRECTORY" --device cuda --num-workers 2
```

Original records contain absolute Colab paths. Restore the original directory layout
for exact replay; do not edit the frozen JSON because that invalidates its digest.
For ordinary inspection, use the saved final outputs once exported; repeated inference
is unnecessary. The evaluator refuses to overwrite a completed output directory.

## Before Task 3

Preserve the source split, ImageNet initialization, official V3 ERM checkpoint, approved
preprocessing/optimizer, and shared MMD definition including its feature treatment.
Confirm Task 3's precise assignment requirements and approve any remaining choices before
coding. Do not choose Task 3 settings using the observed Sketch scores. Keep Task 3 outputs
under a separate root so no Task 1/2 artifact is overwritten.
