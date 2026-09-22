# Task 2 — PACS unsupervised domain adaptation

Technical reproduction and evidence documentation. The student's PDF report text and interpretation are written independently, as required by the assignment.

## Inspect the completed experiment

No GPU, dataset, or trained checkpoint is needed to inspect the published small results or run the saved-evidence check:

```bash
python -m task2.verify_saved
```

This verifies file checksums, exact seed-6304 source split reconstruction, saved best-epoch/early-stopping decisions, and metrics recomputed from source/target/probe prediction tables. It does not retrain a model or repeat image inference.

| Evidence | Location |
|---|---|
| Source train/validation assignments and target IDs | [shared split](../shared/splits/pacs_sketch_seed6304.json) |
| Main comparison and controlled study | [comparison.csv](results/final/comparison.csv): main rows `source_only`, `dan_1`, `dann`, `cdan`; study rows `dan_0p1`, `dan_1`, `dan_10` |
| Target predictions, per-class changes, confusions, probe summaries | [final results](results/final/) |
| Six histories, selected-source metrics and run configurations | [training records](results/training/) |
| Original curves and strength plot | [original plots](results/plots/original/) |
| Clearer full DANN and CDAN histories | [DANN](results/plots/dann_training_clearer.png), [CDAN](results/plots/cdan_training_clearer.png) |
| Fixed-checkpoint verification and diagnostic convergence | [verification summary](results/verification/summary.json), [probe checks](results/verification/probe_checks.json) |
| Source predictions and held-out probe predictions/selection | [verification tables](results/verification/) |
| Original expectations and subsequent correction | [original JSON](provenance/dan_strength_preregistration.json), [dated correction](provenance/PREREGISTRATION-CORRECTION.md) |
| Dataset source, image content hashes, excluded interrupted attempt | [provenance](provenance/) |
| Original, restart and verification environments | [environment](environment/) |

Accuracy/F1 values are fractions in `[0,1]`; accuracy changes are differences of these fractions. Multiply changes by 100 to express percentage points. Domain separability is balanced **source versus target**, with 50% chance. It is distinct from training discriminator accuracy. A dominant-confusion row with `total_errors=0` denotes no errors; its class-name field should not be interpreted as an observed confusion.

The clearer plots retain every epoch. Logarithmic loss axes and a separate domain-accuracy panel improve visibility without changing the data. The selected epoch is marked by a dashed line.

## Fixed protocol

- Sources: Photo, Art Painting, Cartoon. Target: Sketch.
- Per-source stratified 80/20 split with seed 6304: 1336/334 Photo, 1638/410 Art, 1875/469 Cartoon. The same saved source split and source-only checkpoint are to be reused unchanged for Task 3.
- Task 2 permits all 3,929 Sketch images as unlabeled adaptation examples. The training loader supplies images and opaque IDs; it never parses the target class from the path. Paths retain the archive's class folders. Only final evaluation parses those target labels after all six configurations/checkpoints are fixed.
- Torchvision ResNet-18 `IMAGENET1K_V1`, seven-class linear head, complete network fine-tuning. Resize 256×256; random 224 crop and horizontal flip during training; 224 center crop for evaluation; ImageNet normalization.
- All BatchNorm running statistics stay at ImageNet values; affine parameters remain trainable. AdamW learning rate/weight decay both `1e-4`; at most 30 source epochs; patience 5; checkpoint selected by mean macro-F1 across the three source-validation domains.
- Adaptation updates: eight images per source plus 24 unlabeled target images. Shorter loaders cycle. Initialization, sampling, augmentation, optimizer and budget are shared.
- Main methods: source-only, DAN λ=1, DANN, CDAN. Controlled study: DAN λ ∈ {0.1, 1, 10}. No target-label result is used to revise settings.
- DAN uses a sum of three RBF kernels on 512-dimensional features. Exact implemented convention: `exp(-squared_distance / (factor * median_positive_off_diagonal_squared_distance))`, factors `.5, 1, 2`. It uses the biased squared empirical MMD including diagonal kernel terms; the bandwidth median is detached. This documents the completed experiment precisely.
- DANN uses 512→256→2 with ReLU/dropout .5 and scheduled gradient reversal. CDAN uses the 3,584-dimensional feature–probability outer product, without detaching either input or adding entropy conditioning.
- Final domain probe: equal pooled-source-validation and target counts (1,002 each), seed-6304 stratified 70/30 split (1,402/602), balanced logistic regression `C=1`, `max_iter=2000`.

## Dataset and environment

The PACS archive used by this experiment is the copy referenced by [Dassl.pytorch's PACS dataset implementation](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/datasets/dg/pacs.py), Google Drive file ID `1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE`:
[download source](https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE).

Archive SHA256: `0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102`.
Verify this hash before extraction. Extract so that Photo/Art/Cartoon/Sketch folders are under a root such as `/content/atml_pacs/pacs/images`. The original download service may require browser access. Do not substitute another dataset snapshot silently.

The recorded environment was Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.1.3, scikit-learn 1.6.1, Pillow 11.3.0, CUDA 12.8 and cuDNN 91900. Use the three saved environment records. [requirements.txt](requirements.txt) pins the recorded Python packages where versions are known; it is not a complete environment lock. Original SciPy/matplotlib versions were not captured. The fixed-checkpoint probe refits converged and reproduced the original metrics. Hardware/software differences can prevent bitwise reproduction; installing Python packages alone does not recreate a Colab runtime.

In a matching Linux/Colab environment, install missing dependencies if needed:

```bash
python -m pip install -r task2/requirements.txt
```

Do not replace the repository's Task 1 dependency file with this Task 2 file.

## Reproduce the complete experiment in a new location

Run from the repository root. In Colab, put the following shell commands in a `%%bash` cell, first changing to the cloned repository directory.

Preflight only (no training):

```bash
python -m task2.reproduce \
  --pacs-root /content/atml_pacs/pacs/images \
  --output /content/drive/MyDrive/ATML-PA1/task2_reproduction_new
```

To intentionally reproduce the full experiment, add `--execute` to that command. It requires a new output directory outside the repository, verifies the saved split and recorded environment, then trains the six fixed configurations through the original common training code. It freezes all six selected checkpoints before running any target-label evaluation, then produces final tables and curves. The historical submission outputs are not overwritten. This command is for reproduction; it is not required to inspect or accept the already-verified runs.

To regenerate a split for inspection, use a NEW output path and compare it against the submitted manifest before training:

```bash
python -m shared.make_pacs_protocol \
  --pacs-root /content/atml_pacs/pacs/images \
  --output /content/pacs_split_check.json
```

For an interrupted reproduction, preserve its directories and use the original trainer's `--resume` option for the affected run; see `python -m task2.train --help`. Do not start another training run against an existing directory without that option. Complete every fixed configuration before freeze/evaluation. The original uploaded training ZIP is retained in provenance for matching training-code hashes even when a reproduction uses downloaded files without a Git checkout.

## Historical paths and provenance

Completed runs lived under `/content/drive/MyDrive/ATML-PA1/task2_runs`: `source_only`, `dan_1`, `dann`, `cdan`, `t4_restart/dan_0p1`, `t4_restart/dan_10`. Metadata copies are organized here under `results/training/<run_id>`. Historical paths inside JSON are preserved, so the original freeze manifest is evidence and is not directly portable to another machine.

The original interrupted `dan_0p1` attempt is retained under `provenance/excluded_interrupted_dan_0p1`; it was excluded from final evaluation. The fresh T4 run reproduced its first nine saved epochs exactly. Original training ZIP SHA256: `d0ec51f969d16e72a318a811550f193fd91498a2cad26e1735c4de9e7a4aa02c`. Original freeze SHA256: `454100e50843ed28f34ee702f348d2209e88bda59317d4b361f3a7b29e72dcf1`.

The Colab runs used an uploaded ZIP, so their recorded Git commit is null. The corrected freeze gate matches training files against that original ZIP instead. No training file changed. New repository documentation, audit utilities and reproduction entrypoints were added after the experiment; their existence does not imply the original runs executed from this later Git commit.

Verification CSVs combine the six original exports with an added `run_id` column. Probe check JSONs are grouped by run ID. Packaging details and the source evidence ZIP hashes are in [packaging.json](provenance/packaging.json). Large model checkpoints, raw images, and diagnostic feature arrays remain in external artifacts; no fresh Git clone is claimed to include them. A full reproduction rebuilds those artifacts from the fixed dataset/protocol.

## Code sources and assistance

- Torchvision supplies the ResNet-18 architecture, ImageNet weights, and transforms; PyTorch supplies training/autograd. The code implementing the shared loop, MMD and gradient reversal is retained exactly as executed in the uploaded ZIP.
- scikit-learn supplies the stratified split, classification metrics and logistic-regression probe. NumPy, Pillow and matplotlib supply array, image and plotting utilities.
- Dassl.pytorch is credited for the PACS archive reference. Its supplied train/cross-validation split files were not used; the assignment's seed-6304 source split was generated explicitly.
- Codex assisted with implementation, technical auditing, packaging and repository documentation. The student is responsible for understanding submitted code. This documentation and the dated technical correction are not text for the PDF report, whose language and interpretation must be student-authored.
