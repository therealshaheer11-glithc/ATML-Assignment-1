# Task 4 Protocol and Approved Decisions

## Assignment-locked protocol

- Known data: all ten CIFAR-10 classes.
- Split: stratified 90/10 split of the official training partition, seed 6304.
- Optimization uses only the 90% training partition.
- Checkpoint selection uses CIFAR-10 validation accuracy only.
- Final known evaluation uses the complete CIFAR-10 test partition.
- Unknown data: only the fixed 800-image near and 800-image far groups from the
  CIFAR-100 test partition listed in the PA.
- CIFAR-100 training data is never used.
- Backbone: torchvision ResNet-18 from random initialization, with a 3x3 stride-1
  first convolution and no initial max-pooling.
- Vanilla and GCSC: 100 epochs, SGD learning rate 0.1, momentum 0.9, weight decay
  0.0005, cosine decay, batch size 128.
- GCSC changes only the training augmentation by inserting
  `RandAugment(num_ops=2, magnitude=9)` after crop and flip.
- PROSER starts from the selected Vanilla checkpoint, adds five dummy classifiers,
  and fine-tunes the complete network for 50 epochs with learning rate 0.001.
- PROSER uses beta=1 for classifier placeholders, gamma=0.1 for data placeholders,
  `Beta(2,2)` mixing after layer2, and different-class pairs only.
- Required post-hoc scores are MSP, MLS, Energy, and shared-diagonal Mahalanobis.
- Every threshold is determined from CIFAR-10 validation unknownness only.
- Real unknowns remain inaccessible until every checkpoint, score definition,
  Mahalanobis statistic, and threshold is frozen in `evaluation_lock.json`.
- The optional reciprocal-point extension is excluded. No reciprocal-point training
  or evaluation is implemented.

## Choices where the PA is silent

- CIFAR normalization uses the standard CIFAR-10 channel statistics recorded in the
  configuration and code.
- A strictly higher validation accuracy replaces the current best checkpoint. Exact
  ties keep the earlier checkpoint.
- Mahalanobis uses the population mean of within-class squared residuals for the
  shared diagonal variance, followed by the PA-required `1e-6` addition.
- The 95th percentile uses NumPy's linear quantile definition.
- When a different-class permutation is impossible inside a half-batch, a partner
  may be reused; every actual pair still has different labels.
- The PROSER placeholder score follows the authors' delta-probability reference
  score with temperature 1024. The validation-derived dummy bias and final score
  threshold are both frozen before unknown access.
- Failure examples are the three most confidently accepted MLS errors in each
  unknown group, ordered deterministically by score and dataset index.

## External attribution

The PROSER loss and placeholder score are clean-room implementations of the equations
and algorithm in Zhou, Ye, and Zhan, "Learning Placeholders for Open-Set Recognition,"
CVPR 2021, informed by the authors' public reference implementation:
https://github.com/zhoudw-zdw/CVPR21-Proser

The ResNet-18 implementation is provided by torchvision and modified only at the
input stem as required by the PA.

