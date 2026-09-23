# Task 2 assignment-mandated protocol

Source: `ATML-PA1.pdf`, principally PDF pages 5-8, plus general submission rules on
pages 1 and 17.

## Authorship and submission rules

- This is an individual submission; the student's code, experiments, results,
  analysis, and report must be their own.
- Generative AI may assist with coding, but the student is responsible for every
  submitted line and must understand it.
- Generative AI may not write any part of the PDF report. Its language,
  interpretation, and analysis must be entirely the student's own.
- Materially reused external code must be attributed in the README.
- All submitted code must be in a public GitHub repository. The final deliverables
  include an eight-page NeurIPS-style PDF report and the accessible repository link.
- Preserve seeds, configurations, environment information, and machine-readable
  results. Do not commit raw datasets or unnecessary large checkpoints.

## Information protocol

- PACS domains: Photo, Art Painting, Cartoon, and Sketch; seven shared classes.
- Sources: Photo, Art Painting, Cartoon. Target: Sketch.
- All Sketch images may be used without class labels during Task 2 adaptation.
- Sketch labels are available only after all models, settings, and checkpoints are fixed.
- Use seed 6304 and a stratified 80/20 train/validation split within each source domain.
- Optimize with source training labels and select checkpoints by the unweighted mean
  macro-F1 over the three source validation domains.
- Reuse the same source splits in Task 3. The Task 2 Source-only checkpoint is the Task 3
  ERM baseline and must not be retrained differently.
- Task 2 target results may not be used to revise Task 3 settings.

## Model, preprocessing, and optimization

- `torchvision` ResNet-18 with `ResNet18_Weights.IMAGENET1K_V1`.
- Replace the ImageNet classifier with a seven-class linear head and fine-tune the full
  network for every method.
- Resize to 256 by 256; random 224 crop and horizontal flip for training; center 224 crop
  for validation/evaluation; pretrained-weight normalization.
- Freeze BatchNorm running means and variances at their ImageNet values for every method;
  keep gamma and beta trainable. After `model.train()`, set only BatchNorm modules to
  evaluation mode.
- AdamW, learning rate `1e-4`, weight decay `1e-4`, at most 30 source epochs, patience
  five on mean source-validation macro-F1, seed 6304.
- Adaptation updates contain eight images from each source domain and 24 target images.
  Cycle loaders as necessary. Keep initialization, source sampling, augmentation,
  optimizer, budget, and pipeline fixed across methods.

## Required methods

- Source-only: cross-entropy on domain-balanced labeled source batches.
- DAN: MMD on the 512-dimensional pre-head feature, main `lambda_MMD=1`, and a sum of
  three RBF kernels with bandwidths 0.5, 1, and 2 times the median pairwise squared
  feature distance in the current combined batch.
- DANN: `512 -> 256 -> 2` discriminator, ReLU, dropout 0.5, prescribed GRL schedule,
  unit domain-loss weight. Only sources contribute class loss; sources and target
  contribute domain loss.
- CDAN: discriminator input `vec(feature outer-product softmax_probability)` of width
  3584; otherwise use DANN's discriminator settings, schedule, and loss weight. Do not
  detach feature/probability inputs and do not use entropy conditioning.
- Controlled study: DAN lambda in `{0.1,1,10}`. Keep every other setting fixed and state
  expectations before interpreting results.

## Fixed evaluation

- After checkpoint freezing, report each source validation accuracy/macro-F1, their
  means, target accuracy/macro-F1, and target accuracy change from Source-only.
- Domain probe: frozen features, equal pooled-source-validation and target counts, seed
  6304, 70/30 split, balanced logistic regression with `C=1`; chance is 50%.
- Only at final analysis use target labels for per-class accuracy changes and dominant
  confusions.
- Required evidence: the complete method table, classification and alignment/domain-loss
  curves, class-level changes and selected confusions/failures, and a compact alignment-
  strength table or plot.
- Public repository: code, configurations, split logic/indices, environment,
  machine-readable results, and top-level reproduction instructions. Exclude raw data and
  unnecessary large checkpoints.
