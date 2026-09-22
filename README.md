# ATML Programming Assignment 1

Technical documentation for Task 1: inductive biases and feature representations on STL-10. This README covers the saved experiment, its code, dependencies, result files, and reproduction procedure. Report analysis and interpretation are written separately by the student.

Repository: [ATML-Assignment-1](https://github.com/therealshaheer11-glithc/ATML-Assignment-1).

## Inspect the saved evidence

Small result files are under `task1/results/`. Reading these files does not require downloading model weights or running inference.

| Evidence | Files |
| --- | --- |
| Experiment settings and hypotheses | [task1.yaml](task1/configs/task1.yaml) |
| Train, validation, and evaluation identifiers | [split manifests](task1/results/splits/) |
| Training histories and selected epochs | [training records](task1/results/training/) |
| Clean accuracy, macro-F1, confidence, and predictions | [clean results](task1/results/clean/) |
| Grayscale and hue-rotation results | [color results](task1/results/color/) |
| Clean/color/patch compact comparison | [PNG](task1/results/comparison/compact_comparison.png), [CSV](task1/results/comparison/compact_comparison.csv), [record](task1/results/comparison/compact_comparison_record.json) |
| Cue selection, rejection records, counts, and predictions | [cue-conflict results](task1/results/cue_conflict/) |
| Three selected cue-conflict examples | [PNG](task1/results/cue_conflict/figures/selected_examples.png), [predictions](task1/results/cue_conflict/figures/selected_predictions.csv) |
| Translation curves | [final PNG](task1/results/translation/figures/translation_curves_final.png), [CSV](task1/results/translation/figures/translation_curves.csv) |
| Translation directions and image hashes | [directional results](task1/results/translation/), [input manifests](task1/results/translation/inputs/) |
| Patch permutations and predictions | [patch-shuffle results](task1/results/patch_shuffle/) |
| Paired cosine stability | [summaries and per-pair CSVs](task1/results/feature_similarity/) |
| Fixed UMAP selection and coordinates | [selection](task1/results/representation/umap_selection.csv), [coordinates, figures, and records](task1/results/representation/umap/) |

The final translation PNG changes layout only. Its original PNG, curve CSV, and provenance record are retained. The compact comparison uses percentages for accuracy, macro-F1, confidence, and consistency; accuracy changes are in **percentage points**. The original evaluation JSON files generally use fractions for accuracy and consistency. Shape bias and coverage are stored as percentages.

## Fixed experimental settings

| Component | Recorded setting |
| --- | --- |
| Dataset | Official labeled STL-10 train and test partitions; unlabeled partition unused |
| Split | Stratified 80/20 split of the 5,000 training images: 4,000 head-training and 1,000 validation images |
| Evaluation | 500 official test images, 50 per class; saved official indices |
| Random seed | 6304 for splits, head training, deterministic patch permutations, and visualization selection |
| Common image | RGB, direct bicubic resize to 224 × 224; interventions precede model normalization |
| ResNet | torchvision ResNet-50, `ResNet50_Weights.IMAGENET1K_V2`; global-average-pooled 2,048-dimensional feature |
| ViT | torchvision ViT-B/16, `ViT_B_16_Weights.IMAGENET1K_V1`; final 768-dimensional class token |
| CLIP | OpenCLIP `ViT-B-32`, `pretrained="openai"`, `force_quick_gelu=True`; L2-normalized 512-dimensional image embedding |
| Backbones | Frozen parameters and evaluation mode |
| Feature extraction | Batch size 64 for head-training feature extraction; no gradients |
| Heads | One linear classifier per backbone, cross-entropy, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 256 |
| Stopping | At most 50 epochs; stop after five epochs without improved validation accuracy; retain the earliest highest-accuracy validation checkpoint |
| CLIP zero-shot | Fixed prompt `a photo of a {class}.`; normalized image/text embeddings and learned `logit_scale.exp()`; no prompt search |
| Color | Three-channel grayscale and fixed +90° HSV hue rotation |
| AdaIN | Style strength `alpha=0.8`; five unordered class pairs; both directions |
| Translation | 0, 8, 16, and 32 pixels; reflection padding and shifted crop; equal mean over up/down/left/right at nonzero displacement |
| Patch shuffle | 4 × 4 grid of 56 × 56 pixel patches; one non-identity permutation per image, using `SeedSequence([6304, official_index])` |
| Cosine stability | Paired clean/transformed pre-head representations; cue conflicts use their clean content images |
| UMAP | Same 200 unique content images from the accepted cue set, 20 per class; one combined 1,000-point fit per backbone |
| UMAP settings | Five conditions: clean, grayscale, cue conflict, translation at 32 pixels, patch shuffle; `n_neighbors=15`, `min_dist=0.1`, cosine metric, two dimensions, `random_state=6304`, `transform_seed=6304` |

For UMAP, each image has one translation direction, assigned reproducibly with five images per class per direction. Class uses color; clean/transformed condition uses marker style. Coordinates from separately fitted backbones are separate coordinate systems. The CLIP head and zero-shot classifier share one image representation and therefore one CLIP UMAP fit.

The configuration records the protocol. Most experiment scripts implement these settings as constants rather than loading all values dynamically from YAML; editing YAML alone does not change the experiment.

## Environment and storage

[colab-environment.json](colab-environment.json) records the initial Colab environment, including Python, PyTorch, torchvision, package versions, CUDA, and GPU. [requirements-colab.txt](requirements-colab.txt) specifies the added packages `open_clip_torch==3.3.0` and `umap-learn==0.5.12`; it is an add-on list for Colab, not a complete lockfile for an empty Python environment. Later CPU stages record device/software information in their result metadata where available. Use those per-stage records alongside the initial environment file.

Large artifacts are stored outside Git:

| Location | Contents |
| --- | --- |
| `/content/ATML-Assignment-1` | Colab Git working tree |
| `/content/drive/MyDrive/ATML-Assignment-1/artifacts/task1` | Persistent Task 1 artifact root |
| Artifact root: `datasets/stl10` | Official dataset download |
| Artifact root: `checkpoints/linear_heads` | Trained linear heads |
| Artifact root: `features` and `predictions` | Cached tensor outputs |
| Artifact root: `cue_conflict/candidates` | Generated stylized images |
| Artifact root: `cue_conflict/manifests` | Generation, manual review, and frozen selection records |
| Artifact root: `external/pytorch-AdaIN-v0.0.0` | Pinned AdaIN checkout and downloaded weights |
| Artifact root: `metrics`, `representation`, and `figures` | Small numerical results and figures, also copied into the repository |

Raw datasets, generated image collections, model weights, feature tensors, and prediction tensors are excluded from Git. The selected example figure is included as result evidence. A fresh clone alone contains the small evidence files; commands that need images, trained heads, or feature caches also need the corresponding external artifacts.

Some recorded paths are absolute paths from the original Colab run. Preserve historical records; their checksums identify the files used. GPU/CPU and dependency changes can affect bitwise outputs even with fixed seeds. Several scripts reject mismatched cached files or frozen outputs instead of silently replacing them.

The following command examples use Bash. In Colab, use a `%%bash` cell and include these path definitions in the same cell as the commands being run:

```bash
export ATML_REPO=/content/ATML-Assignment-1
export ATML_ARTIFACTS=/content/drive/MyDrive/ATML-Assignment-1/artifacts/task1
export ATML_DATA="$ATML_ARTIFACTS/datasets/stl10"
cd "$ATML_REPO"
```

For a Colab environment that needs the extra packages:

```bash
python -m pip install -r requirements-colab.txt
```

## Build presentation outputs from existing results

These two scripts read saved numerical results and do not run neural networks:

```bash
python task1/analysis/build_compact_comparison.py \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS"

python task1/analysis/format_translation_figure.py \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS"
```

The comparison script checks the saved clean/color/patch predictions before writing its CSV, PNG, and checksum record. The formatting script requires the original translation CSV, PNG, original plotting script, and source results to match their recorded hashes; it creates `translation_curves_final.png` and a new record.

The original translation figure is produced by `task1/analysis/plot_translation.py`. The three-example figure is produced by `task1/analysis/make_cue_examples.py`; it reads saved predictions, the frozen cue manifest, candidate PNGs, and the official test images. It does not repeat classifier inference.

```bash
python task1/analysis/plot_translation.py \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS"

python task1/analysis/make_cue_examples.py \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS"
```

An existing output that differs causes these plotting scripts to stop. To inspect the completed run, use the saved figures directly. Preserve the original plotting script because its hash is part of the original translation provenance.

## Experimental command reference

The following commands rebuild experiments and can write checkpoints or results. They are not needed for the final audit of the completed run. Use a separate working copy and fresh output locations for a new run, preserving the published results first. Scripts also write under `--repo-root/task1/results`; a new artifact directory by itself does not isolate those repository outputs.

### 1. Dataset, splits, heads, and clean/color evaluation

Download the official STL-10 data and generate split manifests:

```bash
python task1/data/make_subset.py \
  --data-root "$ATML_DATA" \
  --output-dir "$ATML_REPO/task1/results/splits" \
  --seed 6304 --evaluation-size 500 --download
```

Train each head on its frozen features. This command trains the head when called, even if valid feature caches already exist:

```bash
for ATML_BACKBONE in resnet50 vit_b_16 clip_vit_b_32; do
  python task1/analysis/train_heads.py \
    --backbone "$ATML_BACKBONE" \
    --repo-root "$ATML_REPO" --data-root "$ATML_DATA" \
    --artifacts-root "$ATML_ARTIFACTS" --device cuda
done

python task1/analysis/evaluate_clean.py \
  --repo-root "$ATML_REPO" --data-root "$ATML_DATA" \
  --artifacts-root "$ATML_ARTIFACTS" --device cuda

python task1/analysis/evaluate_color.py \
  --repo-root "$ATML_REPO" --data-root "$ATML_DATA" \
  --artifacts-root "$ATML_ARTIFACTS" --device cuda
```

The trained CLIP head and zero-shot CLIP are both evaluated by the clean/color scripts. Zero-shot confidence is the maximum softmax probability over scaled class similarities. Training histories and summaries are saved under `metrics/training` in the artifact root; their small copies are packaged under `task1/results/training`.

### 2. AdaIN generation, manual review, and cue evaluation

Class pairs are airplane/cat, car/deer, ship/dog, truck/horse, and bird/monkey. The deterministic schedule contains both directions and 50 matched ranks per pair. Initial generation uses ranks 1–20 in each direction. Replacement generation takes the next unused ascending ranks only in deficient directions.

```bash
python task1/adain/setup_adain.py \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS"

python task1/adain/generate_conflicts.py --mode prepare \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS"

python task1/adain/generate_conflicts.py --mode generate-initial \
  --repo-root "$ATML_REPO" --data-root "$ATML_DATA" \
  --artifacts-root "$ATML_ARTIFACTS" --device cuda
```

Review candidates visually before running classifiers. Reject if the content object is no longer recognizable, is severely obscured or structurally corrupted, has blank regions or major rendering artifacts, or has no perceptible style/texture transfer. Model predictions must not determine retention.

In Colab, launch each review stage in the notebook kernel with `%run` so the widget callbacks remain active. Use the same command after each corresponding generation stage, changing `--scope` to `round-1` and then `round-2`:

```python
%run /content/ATML-Assignment-1/task1/adain/review_candidates.py \
  --artifacts-root /content/drive/MyDrive/ATML-Assignment-1/artifacts/task1 \
  --scope initial --reviewer "REVIEWER NAME"
```

The manual review input is `cue_conflict/manifests/review_manifest.csv` in the artifact root. Keep generation fields and hashes intact. Record `review_status` as `accepted` or `rejected`, an applicable `rejection_reason` for rejected rows, `reviewed_by`, and the actual pre-evaluation review status in `review_completed_before_model_evaluation`. Human review is an explicit input to this workflow; the completed run used interactive Colab cells to record these decisions.

After a complete review, plan and generate replacements. Execute one round, review its new candidates, and only then plan the next round:

```bash
python task1/adain/generate_replacements.py --mode plan --round 1 \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS" --device cpu

python task1/adain/generate_replacements.py --mode generate --round 1 \
  --repo-root "$ATML_REPO" --data-root "$ATML_DATA" \
  --artifacts-root "$ATML_ARTIFACTS" --device cpu --batch-size 4
```

The completed run used 200 initial candidates, 56 round-1 replacements, and 9 round-2 replacements. It reviewed 265 candidates and froze 200 accepted and 65 rejected, with 20 accepted per direction and 40 per unordered pair. Both replacement rounds used CPU. Review snapshots, plans, generation records, and hashes are preserved in `task1/results/cue_conflict/`.

Before cue evaluation, the artifact manifest directory must contain the frozen `accepted_conflicts.csv`, `final_review_manifest.csv`, and `selection_record.json`. For the completed run, these are the recorded human decisions and must be reused unchanged with the matching candidate PNGs. They are copied into the repository for inspection. Freezing was performed in a Colab cell, separately from the generation and evaluation scripts. Its procedure was:

1. Verify all reviews are complete, all candidate PNGs are RGB 224 × 224 images with their recorded hashes, and no cue-classifier results exist yet.
2. Verify 200 accepted rows, 20 per direction, and 40 per pair; preserve the complete review CSV byte-for-byte as `final_review_manifest.csv`.
3. Filter accepted rows, retain the original columns, and sort by numeric `pair_number`, then `direction_code`, then numeric `schedule_rank`. Write `accepted_conflicts.csv` using Python CSV conventions with `newline=""`.
4. Record candidate/accepted/rejected totals, per-direction/per-pair counts, and SHA-256 hashes of both frozen CSVs in `selection_record.json`. Record the actual generation devices and that model predictions were not used for selection. Preserve the existing frozen files for the completed experiment.

The standalone freezer implements those checks and refuses to freeze after cue-classifier outputs exist. For a fresh reproduction, run `--mode freeze` before evaluation. To verify the completed run without modifying it:

```bash
python task1/adain/freeze_selection.py --mode verify \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS"
```

With the frozen selection and candidate images available:

```bash
python task1/analysis/evaluate_cue_conflict.py \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS" \
  --backbone all --device cpu --batch-size 8
```

Predictions are classified as shape, texture, or other. The recorded metrics are `100 * Nshape / (Nshape + Ntexture)` for shape bias and `100 * (Nshape + Ntexture) / Ntotal` for coverage. The three figure IDs are `p02_r01_a_to_b`, `p01_r19_a_to_b`, and `p02_r09_a_to_b`. Figure-example selection is separate from freezing the evaluation set; `browse_cue_examples.py` supports inspection of already-saved predictions after evaluation.

### 3. Translation and patch shuffle

Each translation command evaluates all four cardinal directions for one backbone and displacement. The zero-displacement point is the saved clean identity baseline; four duplicate zero-displacement inference runs are unnecessary.

```bash
for ATML_BACKBONE in resnet50 vit_b_16 clip_vit_b_32; do
  for ATML_DISTANCE in 8 16 32; do
    python task1/analysis/evaluate_translation.py \
      --repo-root "$ATML_REPO" --data-root "$ATML_DATA" \
      --artifacts-root "$ATML_ARTIFACTS" --backbone "$ATML_BACKBONE" \
      --displacement "$ATML_DISTANCE" --device cpu --batch-size 8
  done
done

python task1/analysis/evaluate_patch_shuffle.py --mode prepare \
  --repo-root "$ATML_REPO" --data-root "$ATML_DATA" \
  --artifacts-root "$ATML_ARTIFACTS"

for ATML_BACKBONE in resnet50 vit_b_16 clip_vit_b_32; do
  python task1/analysis/evaluate_patch_shuffle.py --mode evaluate \
    --repo-root "$ATML_REPO" --data-root "$ATML_DATA" \
    --artifacts-root "$ATML_ARTIFACTS" --backbone "$ATML_BACKBONE" \
    --device cpu --batch-size 8
done
```

Image/permutation manifests and hashes allow the transformed inputs to be checked across predictors. Prediction consistency always compares each transformed prediction with that same predictor's clean prediction on the corresponding image.

### 4. Representation analysis from saved features

These commands consume feature/prediction caches from the preceding stages. They do not load pretrained networks or regenerate features. UMAP fitting itself is additional analysis computation; use the saved coordinates and figures when only inspecting the completed run.

```bash
for ATML_BACKBONE in resnet50 vit_b_16 clip_vit_b_32; do
  python task1/analysis/feature_similarity.py \
    --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS" \
    --backbone "$ATML_BACKBONE"
done

python task1/analysis/representation_umap.py --mode prepare \
  --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS"

for ATML_BACKBONE in resnet50 vit_b_16 clip_vit_b_32; do
  python task1/analysis/representation_umap.py --mode plot \
    --repo-root "$ATML_REPO" --artifacts-root "$ATML_ARTIFACTS" \
    --backbone "$ATML_BACKBONE"
done
```

Cosine summaries cover grayscale, cue conflict, patch shuffle, and all 8/16/32-pixel translations in four directions. Translation aggregates contain 2,000 image-direction pairs per displacement. UMAP records include feature-cache hashes, subset hashes, parameters, software versions, coordinate CSV hashes, and figure hashes.

## External code and model attribution

### Materially reused AdaIN implementation

The method is **Xun Huang and Serge Belongie, “Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization,” ICCV 2017** ([paper](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html)). The authors' original Torch7 implementation is [xunhuang1995/AdaIN-style](https://github.com/xunhuang1995/AdaIN-style).

The executable code used here is **Naoto Inoue's unofficial PyTorch reproduction**, [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN), release **v0.0.0**, pinned commit **`324eedefee5e11aa23a8f90cf1d7d87cdcc00860`**. Its license is **MIT**, copyright 2018 Naoto Inoue ([license at the pinned commit](https://github.com/naoto0804/pytorch-AdaIN/blob/324eedefee5e11aa23a8f90cf1d7d87cdcc00860/LICENSE)). The upstream checkout retains its license.

`generate_conflicts.py` loads the upstream `function.py` implementation of adaptive instance normalization and upstream `net.py` VGG/decoder definitions. It loads the release's pretrained normalized VGG and decoder weights, takes the VGG prefix through the required feature layer, and performs alpha blending and decoding. The same loader is reused for replacement generation. Dataset pairing, schedules, review bookkeeping, frozen selections, classifier evaluation, and output packaging are assignment-specific surrounding code. The AdaIN algorithm, networks, and weights are not claimed as original implementation or training.

| Upstream weight | SHA-256 |
| --- | --- |
| `decoder.pth` | `379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472` |
| `vgg_normalised.pth` | `804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a` |

Release download URLs, upstream source-file hashes, license-file hash, file sizes, and the resolved commit are recorded in [adain_source.json](task1/results/cue_conflict/adain_source.json). `setup_adain.py` obtains that checkout and the release weights; large external assets are stored outside Git.

### Libraries and pretrained backbones

PyTorch and torchvision supply the neural-network components, STL-10 loader, transformations, and pretrained ResNet/ViT models. OpenCLIP supplies the OpenAI-pretrained CLIP image/text model and tokenizer. NumPy and scikit-learn support deterministic sampling and splits; `umap-learn` supplies UMAP; Pillow and Matplotlib support image handling and figures. Versions used are recorded in the environment and per-stage result files.

### Coding assistance

ChatGPT/Codex assisted with code generation, debugging, result-file checks, figure formatting, and this technical README. The student is responsible for understanding and verifying submitted code. Under the assignment's AI rule, PDF-report language, interpretation, and analysis must be written entirely by the student.
