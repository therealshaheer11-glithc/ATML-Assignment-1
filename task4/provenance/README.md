# Task 4 Provenance

This directory contains the records used to authenticate the completed Task 4 experiment.

## Files

- [`runtime_preflight.json`](runtime_preflight.json): recorded software, CUDA, and GPU environment.
- [`cifar10_download.json`](cifar10_download.json): CIFAR-10 download and archive provenance.
- [`cifar100_final_evaluation_download.json`](cifar100_final_evaluation_download.json): provenance for the authorized final CIFAR-100 evaluation download.
- [`cifar10_seed6304.json`](cifar10_seed6304.json): fixed CIFAR-10 training and validation split.
- [`evaluation_lock.json`](evaluation_lock.json): checkpoint, configuration, score, threshold, and unknown-group choices frozen before CIFAR-100 access.
- [`mahalanobis_stats.npz`](mahalanobis_stats.npz): CIFAR-10-only statistics authenticated by the evaluation lock.
- [`artifact_manifest.json`](artifact_manifest.json): file inventory for the original Task 4 publication snapshot.

## Artifact-manifest scope

`artifact_manifest.json` records the original Task 4 publication snapshot at repository commit:

```text
ef2ef5033f197b94d498f077ba423a3bb8660ad1
```

All 21 files listed in the manifest match their recorded byte sizes and SHA-256 identities at that commit.

Later documentation-only commits legitimately changed files such as the root README and the Task 4 README. Their current hashes therefore do not match the historical publication manifest. This does not indicate that the experiment results or locked evaluation artifacts changed.

The historical manifest should not be regenerated merely because documentation is improved. Use the commit above when authenticating the complete original publication package.

## Immutable evaluation identity

The final evaluation-lock SHA-256 is:

```text
d71fee7c64e8efc1f9622bffd08b7bae85fa2227c9a49c1ba4d7af759669f6ac
```

The lock records that:

- training, checkpoint selection, score design, and threshold selection used CIFAR-10 only;
- zero CIFAR-100 images were accessed before the lock;
- RPL was not included;
- the near and far unknown groups were fixed before evaluation; and
- no checkpoint, score, threshold, or calibration value changed after unknown-data access.

## External storage boundary

Large selected model checkpoints and cached final logits and features remain in Google Drive and are intentionally excluded from Git. Their identities are recorded in the evaluation lock and completion records.

The repository contains the fixed split, evaluation lock, training histories, final machine-readable results, comparison tables, failure analysis, figure, and provenance needed to audit the published outcome.
