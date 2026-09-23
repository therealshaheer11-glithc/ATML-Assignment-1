# Pre-training audit matrix

This matrix maps every approved decision to its implementation. `PASS (local)` means a
CPU unit check was executed before packaging. Dataset/GPU checks remain `COLAB GATE`
until the fresh Colab runtime runs the preflight.

| Decision | Implementation | Check |
|---|---|---|
| D1 | `task2/configs/dan_*.json`, `task2/config.py` | Config lock pass |
| D2 | `shared/mmd.py::three_kernel_mmd` | Exact expression PASS (local) |
| D3 | `shared/mmd.py` strict upper triangle | Off-diagonal-zero test PASS (local) |
| D4 | `shared/mmd.py` denominator `2 * bandwidth` | Formula test PASS (local) |
| D5 | `shared/mmd.py` invalid-median exception | Stop test PASS (local) |
| D6 | `shared/pacs.py::CyclingBatchSampler` | Code/config audit; COLAB GATE |
| D7 | `task2/train.py` locked AdamW and mean CE | Code/config audit |
| D8 | `task2/train.py` deterministic FP32, global norm clipping at 20, finite checks | Clipping test PASS (local); COLAB GATE |
| D9 | `task2/prepare_initialization.py` | Hash gate; COLAB GATE |
| D10 | `task2/methods.py`, `task2/train.py` | Schedule PASS (local) |
| D11 | `task2/train.py` source validation selection | Code audit |
| D12 | Frozen-checkpoint evaluation phase | Not run before freeze |
| D13 | `task2/train.py::validate_sources` | Code audit |
| D14 | `shared/pacs.py::image_transform` | Code audit; COLAB GATE |
| D15 | `task2/train.py` epoch checkpoint and identity locks | Code audit; COLAB GATE |
| D16 | `shared/mmd.py::l2_normalize_mmd_features`, DAN branch in `task2/train.py` | Norm/gradient/zero-norm tests PASS (local); COLAB GATE |

The local syntax check and all eight high-value unit checks passed. No training or target
evaluation was performed during packaging.
