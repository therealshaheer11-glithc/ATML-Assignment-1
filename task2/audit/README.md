# Historical Task 2 fixed-checkpoint verification

`verify_frozen_colab.py` is a byte-identical copy of the standalone helper used to produce the 22 September 2026 fixed-checkpoint verification archive and clearer DANN/CDAN plots. Original filename: `ATML-PA1-Task2-verify-frozen.py`.

SHA256: `65e1ba8ec9d7525f0637336bdf9aed85d1bc9c9eb4b29cfed9d8c079e8448c06`.

## Scope

The helper checks original code and checkpoint hashes, compares BatchNorm buffers with official pretrained tensors, repeats inference from already frozen checkpoints, and refits the original fixed domain probe. It writes a new timestamped verification directory and archive, and generates the clearer training curves. It does not train or select a replacement PACS model.

## Historical Colab prerequisites

- Mounted Google Drive and the original six checkpoints and metadata under `/content/drive/MyDrive/ATML-PA1/task2_runs`.
- Original runtime code under `/content/atml_pa1_task2_code`, including the corrected freeze module.
- Original PACS images under `/content/atml_pacs/pacs/images`.
- The recorded T4/software environment. The helper stops if its checks fail.

Run this file from a Colab cell in that restored environment:

```python
import runpy
runpy.run_path('/content/ATML-PA1-Task2-verify-frozen.py', run_name='__main__')
```

The command assumes the unchanged helper has been uploaded using its original filename. Alternatively pass the path to `task2/audit/verify_frozen_colab.py` in a checkout. Historical absolute paths are intentionally retained so this copy documents the exact verification performed; it is not a portable checkpoint locator.

For a complete experiment reproduction in a new directory, use `task2.reproduce` as documented in the Task 2 README. For checks of the published small results without inference, use `task2.verify_saved`.

This helper is included for audit and figure reproducibility. Codex assisted with implementation and technical documentation; the student's assignment report must be written independently.
