# Task 2 implementation notes

Added retrospectively on **23 September 2026**, after training and target evaluation. This file documents the completed implementation. It is not a preregistration, course-staff approval, or assignment-report text.

## Relationship to the assignment

Assignment pages 6–8 prescribe the main architecture, domains, split ratio/seed, transforms, BatchNorm policy, optimizer/rates, batch composition, stopping rule, objectives, study values and diagnostic. The code must also resolve details that those pages leave implicit. This note records the settings actually used, including inherited library defaults.

The technical review found no clear contradiction between the other choices below and an explicit PDF requirement. **The positive-distance MMD median remains an unresolved interpretation issue:** the PDF requests the median pairwise squared feature distance, while the implementation removes all zero distances before computing that median. Removing zero distances between different examples can change the requested statistic. The training histories do not establish whether or how often this affected original batches. This choice should not be described as an unqualified literal match to the PDF. Documentation alone cannot make a conflicting implementation satisfy a requirement.

The PDF also does not write the complete RBF exponent or specify bandwidth gradients and every empirical-estimator detail. The exact completed definition is recorded below so it can be clarified with the teaching team if necessary. No causal explanation for the poor runs is established by this inventory.

## 1. Exact DAN implementation

Evidence: [MMD code](../shared/mmd.py), [base configuration](configs/base.json).

- Concatenate 24 source and 24 target 512-dimensional features; cast to float32; compute squared Euclidean distances with `torch.cdist(...).square()`.
- For bandwidth selection, remove the matrix diagonal and then remove every remaining zero distance. Both pair directions are included.
- Compute the median with PyTorch's lower-middle convention. Detach this value from autograd. Use 1.0 when no positive distances exist; clamp the bandwidth to at least `1e-12`.
- Use `K = sum(exp(-d_squared / (factor * median)))` for factors 0.5, 1 and 2. There is no additional factor of two or second squaring of the median.
- Use biased empirical squared MMD: `mean(Kss) + mean(Ktt) - 2 * mean(Kst)`, including self-pairs in the within-group kernel blocks. The diagonal exclusion for bandwidth selection is separate from diagonal inclusion in the estimator.
- Recompute bandwidth per batch but backpropagate only through kernel distances, not through bandwidth selection. Add lambda times this MMD to source mean cross-entropy.

The estimator and exponent convention were already recorded in the original configuration. The later README also records bandwidth detachment. This note makes their status as implementation conventions explicit. The minimum-bandwidth safeguard already prevents a zero denominator; zero-pair exclusion is an additional statistical choice.

## 2. Epoch, sampling and splits

Evidence: [shared PACS code](../shared/pacs.py), [trainer](train.py), [saved split](../shared/splits/pacs_sketch_seed6304.json).

- PACS comes from the recorded Dassl-referenced archive. Files are sorted; scikit-learn performs an independent stratified 80/20 split per source using seed 6304. Its rounding and allocation rules determine the exact memberships. Selected indices are sorted back into canonical order. The archive's own split text files are not used.
- Define one source epoch as `max(ceil(n_source_train / 8))`: 235 updates. Each epoch draws 1,880 examples per source and 5,640 target examples, cycling when needed. The maximum budget is 7,050 updates.
- Within a domain, shuffle without replacement, reshuffle upon exhaustion, and fill each complete batch across permutation boundaries. Reset loaders each epoch. Sampling balances domains, without adding class-balanced sampling.
- Source seed streams are `6304 + 100000 * domain_index + 10000 * epoch_index`; target uses `6304 + 900000 + 10000 * epoch_index`. Domain order is Photo, Art Painting, Cartoon; indices are zero-based. Use NumPy `default_rng` permutations and a seeded Torch generator per training loader.
- Use two data-loader workers, pinned memory on CUDA, validation/evaluation batch size 64, no evaluation shuffle and a retained final partial evaluation batch. Each training run starts in a separate seeded Python subprocess.

## 3. Preprocessing, initialization and optimization

Evidence: [PACS code](../shared/pacs.py), [model](model.py), [trainer](train.py), [adversarial methods](methods.py).

- Pillow decodes images and converts to RGB. Resize to the required 256×256 using the default bilinear/Pillow antialiasing behavior; apply random 224 crop then horizontal flip with probability 0.5 for both source and adaptation-target training. Evaluation uses the required center crop and ImageNet normalization. Class order is dog, elephant, giraffe, guitar, horse, house, person.
- New classifier/discriminator linear layers retain biases and the default Torch random initialization. Construct the pretrained backbone, replacement head, then discriminator in that order. Saved initial classifier-model hashes match across methods.
- AdamW uses the required learning rate and weight decay of `1e-4`, plus defaults `betas=(0.9, 0.999)`, `eps=1e-8`, `amsgrad=False`. One parameter group applies weight decay to all trainable parameters, including biases and BatchNorm affine parameters. Backend switches such as `foreach` and `fused` retain defaults.
- Keep learning rate constant. Use ordinary float32 training, one backward pass and optimizer step per update, without gradient clipping, accumulation or mixed-precision scaling. A non-finite total loss raises an error; large finite losses remain in the history.
- Concatenate sources in the fixed domain order and append target for one adaptation forward pass. Mean source cross-entropy is unweighted across classes, with no label smoothing. Mean binary-domain cross-entropy covers all 48 examples; source=0 and target=1.
- Update discriminator and classifier/backbone jointly in the same optimizer step. Define GRL progress as `p = (epoch_index * 235 + step_index) / (30 * 235 - 1)`. Early stopping truncates this planned schedule. CDAN uses the exact flattened feature-by-softmax-probability outer product with ordinary softmax temperature and no extra feature normalization.
- Seed Python, NumPy, CPU/CUDA Torch; disable cuDNN benchmarking and enable cuDNN deterministic behavior. The experiment did not enforce every deterministic-algorithm/backend setting, and the historical environment record is not a complete lockfile.

## 4. Checkpoint selection and domain probe

Evidence: [trainer](train.py), [final evaluator](evaluate_final.py), [recorded probe parameters](results/verification/probe_checks.json), [probe selection IDs](results/verification/probe_selection.csv).

- Validate after every complete epoch. A score must exceed the previous best by more than `1e-12`; ties preserve the earlier epoch and count toward patience. There is no epoch-zero checkpoint candidate.
- Use argmax predictions, all seven labels for macro-F1 and `zero_division=0`. Source mean metrics weight each domain equally. Per-class accuracy is recall within the true class.
- For the probe, sample 334 validation images without replacement from each source: 1,002 source examples. Sample 1,002 target examples. Equal representation of the three sources is additional to the required equality between pooled-source and target counts. All 1,213 source-validation images still contribute to classification validation.
- Reinitialize a seed-6304 NumPy stream per model and sample Photo, Art, Cartoon, then target, yielding identical selected IDs across models. Stratify the required 70/30 split by binary domain identity: 1,402 fitting and 602 held-out examples. Object-class labels do not determine probe selection or partition.
- Fit the probe on raw 512-dimensional features without standardization, length normalization or dimensionality reduction. Use `LogisticRegression(C=1, class_weight='balanced', max_iter=2000, random_state=6304)` with inherited `solver='lbfgs'`, `penalty='l2'`, `tol=1e-4`, `fit_intercept=True`. Historical verification records confirm these parameters and convergence.
- Report one held-out accuracy per model. Later verification refits the same diagnostic with the same IDs/settings. The experiment reports one selected seed-6304 run per configuration, not a multi-seed average.

## 5. Evidence handling and presentation

- Epoch losses are means over updates. DAN curves show unweighted MMD; discriminator accuracy is measured in training mode and is distinct from final probe accuracy.
- The controlled study reuses the main lambda=1 run and uses a log lambda axis. Later clearer adversarial figures use log loss axes and separate accuracy panels, retaining all epochs.
- Dominant confusion uses the largest off-diagonal count, with first-class-index tie breaking. Zero-error rows have a zero count and must not be interpreted as actual confusions.
- Best/last checkpoints are written at epoch boundaries; continuation resumes at the next completed epoch. The chosen fresh T4 study runs and excluded interrupted attempt are documented in the existing README.
- Six-run freezing, named-file code hashes, ZIP provenance, JSON configurations and Drive paths are implementation choices for preserving the prescribed evaluation boundary. Later documentation is dated as such. Original predictions, configurations, checkpoints and expectations remain the historical evidence.

## Follow-through

This note supplies technical reproduction details previously spread across code, defaults and metadata. It is an addition to the historical package; the original submission manifest describes that earlier package and does not hash this later note.

The outstanding course-specific question concerns acceptance of the exact MMD bandwidth convention, especially filtering zero distances between different examples. If the teaching team requires a correction, it should follow a documented, consistently applied protocol for affected comparisons, preserve the original evidence and disclose that target results were already seen. Retrospective documentation is not retrospective approval or a new preregistration.

The assignment report's wording and interpretation must be independently written by the student. Codex assisted with this technical documentation.
