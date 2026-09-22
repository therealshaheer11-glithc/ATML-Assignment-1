"""Reproduce the fixed six-run protocol in a NEW output directory.
Default: preflight only. --execute explicitly starts the full experiment.
Historical submitted outputs are never rewritten by this entrypoint.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision

from shared.pacs import make_protocol, resolve_root
from task2.freeze import freeze
from task2.train import MAIN_RUNS

REPO = Path(__file__).resolve().parents[1]
SOURCE = 'Dassl PACS ZIP SHA256:0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pacs-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true', help='Explicitly run all six training configurations, freeze, then evaluate')
    args = parser.parse_args()
    root = resolve_root(args.pacs_root)
    protocol_path = REPO / 'shared/splits/pacs_sketch_seed6304.json'
    saved_protocol = json.loads(protocol_path.read_text())
    if make_protocol(root) != saved_protocol:
        raise RuntimeError('Dataset or regenerated seed-6304 split differs from the submitted protocol')
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError('Choose a NEW output directory outside this repository; existing outputs cannot be reused')
    actual = {'python': platform.python_version(), 'torch': str(torch.__version__),
              'torchvision': torchvision.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__}
    expected = json.loads((REPO / 'task2/environment/t4-restart.json').read_text())
    differences = {key: {'expected': expected[key], 'actual': value} for key, value in actual.items() if expected[key] != value}
    if differences:
        raise RuntimeError(f'Recorded environment differs: {differences}')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This reproduction entrypoint expects the recorded Tesla T4 runtime')
    if torch.version.cuda != expected['cuda'] or torch.backends.cudnn.version() != expected['cudnn']:
        raise RuntimeError('CUDA/cuDNN differs from the recorded T4 runtime')
    print('Preflight passed: dataset, exact split, software and T4 match.')
    if not args.execute:
        print('No experiment started. Add --execute only to reproduce all six runs in the new output directory.')
        return
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, output / 'pacs_sketch_seed6304.json')
    shutil.copy2(REPO / 'task2/provenance/dan_strength_preregistration.json', output / 'original_preregistration.json')
    (output / 'environment.json').write_text(json.dumps({**actual, 'gpu': torch.cuda.get_device_name(0),
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'scope': 'Reproduction of the original fixed protocol; original study expectations copied as historical context'}, indent=2)+'\n')
    # All six models complete before any target labels are evaluated.
    for name in MAIN_RUNS:
        subprocess.run([sys.executable, '-m', 'task2.train', '--run-id', name,
            '--pacs-root', str(root), '--protocol', str(output / 'pacs_sketch_seed6304.json'),
            '--dataset-source', SOURCE, '--output', str(output), '--num-workers', '2'], cwd=REPO, check=True)
    lock_path = output / 'freeze.json'
    freeze({name: output / name for name in MAIN_RUNS}, lock_path,
           code_archive=REPO / 'task2/provenance/original-training-code.zip')
    subprocess.run([sys.executable, '-m', 'task2.evaluate_final', '--pacs-root', str(root),
        '--protocol', str(output / 'pacs_sketch_seed6304.json'), '--freeze', str(lock_path),
        '--output', str(output / 'final'), '--num-workers', '2'], cwd=REPO, check=True)
    subprocess.run([sys.executable, '-m', 'task2.plot_results', '--freeze', str(lock_path),
        '--final-dir', str(output / 'final'), '--output', str(output / 'plots')], cwd=REPO, check=True)
    print('Reproduction complete:', output)

if __name__ == '__main__':
    main()
