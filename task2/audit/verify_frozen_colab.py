"""Verify fixed Task 2 checkpoints; never train or select a replacement model.
Creates a new timestamped verification directory. Original artifacts are read-only.
The logistic probe is refitted once with its original fixed settings.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import torch
import torchvision
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RUNS = Path('/content/drive/MyDrive/ATML-PA1/task2_runs')
CODE = Path('/content/atml_pa1_task2_code')
DATA = Path('/content/atml_pacs/pacs/images')
FROZEN_SHA = '454100e50843ed28f34ee702f348d2209e88bda59317d4b361f3a7b29e72dcf1'
SOURCES = ('photo', 'art_painting', 'cartoon')
EXPECTED_RUNTIME_HASHES = {'shared/__init__.py': '2620e88f617e177e3f69890bc0986b531495c49335e930e051c0d40c146a1caa', 'shared/make_pacs_protocol.py': 'cbe3fa8461002d0db8373abb7ee195bf80cef2f0d7597b0905b50d24a3c89515', 'shared/mmd.py': '35aca0124a05df421943bbe7fb13df09d92e26d441d2e4133bc0e9fd48d52934', 'shared/pacs.py': 'eb9c2101cefb9570f2d20a9611c2ee6ba6122b79d2a12b017c3e05fd0185944b', 'task2/__init__.py': '89198a8614fc3709ede459aa4b1cd9be04bf7786c3b421e64776888205105d8c', 'task2/configs/base.json': '3e73168761cd2088330012e0fffa5ee762fe892b2379ce9b726fb5837b41c06c', 'task2/configs/cdan.json': 'bbd89984f9774661752a569dd2407dc7b8a2dd3676e79fc85b153c5bc86419c4', 'task2/configs/dan_0p1.json': 'cada3a8d3afe7e5c23f7d1d0b21537752ae41c5a80ed948bb5860ade00507aff', 'task2/configs/dan_1.json': '99d548122f5cbcaca6f1d4b7cefaede7298df5bc591d27069da84c6aff16c336', 'task2/configs/dan_10.json': 'befa58a99db1a5d6056aa6c47b8c2f82d496d2a3b2f5eebb0e81b52d0269ee93', 'task2/configs/dann.json': '8cd006da139b2b10a107d60a0b3661b81d1bd9df83af087afbc9b3b9a0c358ad', 'task2/configs/source_only.json': 'ea19f96099a0b83543e43d2007b979344d5c498f8ef4bad39e1b310003c60392', 'task2/evaluate_final.py': '4bb1a6495c2cf87ad1223101ef43e3ccaf3d01d5263c1ce01a9bd7abddd4bf04', 'task2/freeze.py': '5d645aedbc03feb372d947b691c43717d26c964dafd7deca394573d96585c774', 'task2/methods.py': 'e588073bbde832b43c9fcfa5fb3837fc7755e21fee2286984ea0b2554dbdaaa0', 'task2/model.py': '9c3fd922c6dd1fdc28967a07458f4176612130312e273f3b3510275b5e8240b8', 'task2/plot_results.py': '1dd13f8d2aba7946fb09694ba43d31646ab93cce4239ab6e6b7de35bafc529c2', 'task2/train.py': '816bd998ff476fda8c47e08ce02b0dd2bdc8af20d0faeca03b93e293ff808fe3'}

def read_json(path):
    return json.loads(path.read_text())

def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def clearer_plot(history, name, best_epoch, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [int(row['epoch']) for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes.ravel()
    for axis, field, title in [(ax[0], 'classification_loss', 'Source classification loss'),
                               (ax[1], 'domain_loss', 'Domain loss')]:
        axis.plot(epochs, [float(row[field]) for row in history], marker='.')
        axis.set_yscale('log')
        axis.set_title(title + ' (log scale)')
        axis.set_ylabel('Loss')
    ax[2].plot(epochs, [float(row['domain_accuracy']) for row in history], marker='.')
    ax[2].axhline(.5, color='gray', linestyle=':', label='50% chance')
    ax[2].set(title='Training domain accuracy', ylim=(0, 1), ylabel='Accuracy')
    ax[2].legend()
    for domain in SOURCES:
        ax[3].plot(epochs, [float(row[domain + '_macro_f1']) for row in history], label=domain)
    ax[3].plot(epochs, [float(row['mean_source_macro_f1']) for row in history], color='black', linewidth=2, label='Mean')
    ax[3].set(title='Source validation macro-F1', ylim=(0, 1), ylabel='Macro-F1')
    ax[3].legend(fontsize=8)
    for axis in ax:
        axis.axvline(best_epoch, color='gray', linestyle='--')
        axis.set_xlabel('Source epoch')
        axis.grid(alpha=.2)
    fig.suptitle(f'{name.upper()} — all epochs; dashed line = selected epoch {best_epoch}')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

def probe_check(source_features, source_ids, target_features, target_ids, output):
    rng = np.random.default_rng(6304)
    count = min(*(len(source_features[d]) for d in SOURCES), len(target_features) // 3)
    blocks, ids = [], []
    for domain in SOURCES:
        selected = rng.choice(len(source_features[domain]), count, replace=False)
        blocks.append(source_features[domain][selected])
        ids.extend(source_ids[domain][int(i)] for i in selected)
    selected = rng.choice(len(target_features), count * 3, replace=False)
    blocks.append(target_features[selected])
    ids.extend(target_ids[int(i)] for i in selected)
    features = np.concatenate(blocks)
    truth = np.r_[np.zeros(count * 3, dtype=int), np.ones(count * 3, dtype=int)]
    train, test = train_test_split(np.arange(len(truth)), test_size=.3, random_state=6304, stratify=truth)
    probe = LogisticRegression(C=1, class_weight='balanced', max_iter=2000, random_state=6304)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        probe.fit(features[train], truth[train])
    prediction = probe.predict(features[test])
    np.savez_compressed(output / 'probe_features.npz', features=features, domain_labels=truth,
                        train_indices=train, test_indices=test, coefficients=probe.coef_, intercept=probe.intercept_)
    test_set = set(map(int, test))
    write_csv(output / 'probe_selection.csv', [
        {'index': i, 'id': identifier, 'domain_label': int(truth[i]), 'partition': 'test' if i in test_set else 'train'}
        for i, identifier in enumerate(ids)])
    write_csv(output / 'probe_predictions.csv', [
        {'index': int(i), 'truth': int(truth[i]), 'prediction': int(pred)} for i, pred in zip(test, prediction)])
    result = {'accuracy': float(accuracy_score(truth[test], prediction)),
              'iterations': probe.n_iter_.tolist(),
              'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in caught),
              'warnings': [str(w.message) for w in caught],
              'parameters': probe.get_params(), 'source_count': count * 3, 'target_count': count * 3,
              'train_count': len(train), 'test_count': len(test)}
    write_json(output / 'probe_check.json', result)
    return result

def main():
    from google.colab import drive, files
    if not RUNS.exists():
        drive.mount('/content/drive')
    assert (CODE / 'task2/train.py').is_file(), 'Original runtime code is missing. Stop here and share this message; do not retrain.'
    assert DATA.is_dir(), 'Local PACS images are missing. Stop here and share this message; do not retrain.'
    for relative, expected in EXPECTED_RUNTIME_HASHES.items():
        assert sha(CODE / relative) == expected, f'Runtime file changed: {relative}. Stop for inspection.'
    assert torch.cuda.is_available(), 'Select the T4 GPU runtime for this comparison, then rerun this verification cell.'
    gpu = torch.cuda.get_device_name(0)
    assert 'T4' in gpu, f'Expected original T4 hardware; found {gpu}'
    lock_path = RUNS / 'task2_freeze_t4.json'
    assert sha(lock_path) == FROZEN_SHA, 'Freeze file has changed; stop for inspection.'
    lock = read_json(lock_path)
    actual_env = {'python': platform.python_version(), 'torch': str(torch.__version__),
                  'torchvision': torchvision.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__}
    saved_env = read_json(RUNS / 't4_restart/environment.json')
    differences = {key: [saved_env[key], value] for key, value in actual_env.items() if saved_env[key] != value}
    assert not differences, f'Environment differs from original: {differences}. Stop and share this message.'
    assert torch.version.cuda == saved_env['cuda'] and torch.backends.cudnn.version() == saved_env['cudnn'], 'CUDA/cuDNN differs from recorded T4 environment.'
    sys.path.insert(0, str(CODE))
    # Fail rather than use a cached module imported from another code directory.
    for name, module in list(sys.modules.items()):
        if name == 'shared' or name.startswith('shared.') or name == 'task2' or name.startswith('task2.'):
            location = getattr(module, '__file__', None)
            assert location and Path(location).resolve().is_relative_to(CODE.resolve()), f'Unexpected loaded module: {name}'
    from shared.pacs import CLASSES, load_protocol
    from task2.model import PACSClassifier
    from task2.train import make_validation_loader, verify_dataset_snapshot, source_code_sha256
    from task2.evaluate_final import FinalLabeledSketch, extract, validate_lock
    protocol = load_protocol(RUNS / 'pacs_sketch_seed6304.json')
    validate_lock(lock, protocol)
    verify_dataset_snapshot(DATA, protocol)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = RUNS.parent / ('task2_verification_' + stamp)
    output.mkdir(exist_ok=False)
    write_json(output / 'verification_environment.json', {**actual_env, 'gpu': gpu, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'freeze_sha256': sha(lock_path),
        'protocol_sha256': sha(RUNS / 'pacs_sketch_seed6304.json'), 'training_source_sha256': source_code_sha256(),
        'scope': 'Fixed-checkpoint inference and original fixed logistic probe; no model training or reselection'})
    device = torch.device('cuda')
    # Compare checkpoint BatchNorm buffers with official ImageNet weights.
    pretrained = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    bn_reference = {k: v for k, v in pretrained.items() if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    del pretrained
    source_loaders = {d: make_validation_loader(DATA, protocol['source_splits'][d]['validation'], 2, device) for d in SOURCES}
    target_loader = torch.utils.data.DataLoader(FinalLabeledSketch(DATA, protocol['target_unlabeled']), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    prior = {r['run_id']: r for r in read_csv(RUNS / 'task2_final_t4/comparison.csv')}
    prior_predictions = read_csv(RUNS / 'task2_final_t4/target_predictions.csv')
    summary = {}
    for name, record in lock['runs'].items():
        print(f'{name}: checking frozen checkpoint and extracting evaluation features...', flush=True)
        folder = output / name
        folder.mkdir()
        checkpoint_path = Path(record['directory']) / 'best.pt'
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert saved['epoch'] == record['best_epoch'] and saved['config'] == record['config']
        state = saved['model_state']
        bn_match = all(torch.equal(state['backbone.' + k], v) for k, v in bn_reference.items())
        model = PACSClassifier(pretrained=False).to(device)
        model.load_state_dict(state)
        model.eval()
        model.requires_grad_(False)
        metrics, source_features, source_ids, source_predictions = {}, {}, {}, []
        for domain in SOURCES:
            features, logits, truth, ids = extract(model, source_loaders[domain], device)
            prediction = logits.argmax(1)
            source_features[domain], source_ids[domain] = features, ids
            metrics[domain + '_accuracy'] = float(accuracy_score(truth, prediction))
            metrics[domain + '_macro_f1'] = float(f1_score(truth, prediction, labels=list(range(7)), average='macro', zero_division=0))
            source_predictions.extend({'domain': domain, 'path': identifier, 'truth': int(y), 'prediction': int(p)} for identifier, y, p in zip(ids, truth, prediction))
        for field in ('accuracy', 'macro_f1'):
            metrics['mean_source_' + field] = float(np.mean([metrics[d + '_' + field] for d in SOURCES]))
        write_csv(folder / 'source_predictions.csv', source_predictions)
        target_features, target_logits, truth, ids = extract(model, target_loader, device)
        target_pred = target_logits.argmax(1)
        target_same = (ids == [p['id'] for p in prior_predictions] and
                       [CLASSES[int(p)] for p in target_pred] == [p[name + '_prediction'] for p in prior_predictions] and
                       [CLASSES[int(y)] for y in truth] == [p['true_class'] for p in prior_predictions])
        metrics['target_accuracy'] = float(accuracy_score(truth, target_pred))
        metrics['target_macro_f1'] = float(f1_score(truth, target_pred, labels=list(range(7)), average='macro', zero_division=0))
        write_csv(folder / 'target_predictions.csv', [{'id': i, 'truth': int(y), 'prediction': int(p)} for i, y, p in zip(ids, truth, target_pred)])
        print(f'{name}: refitting the original diagnostic probe; model remains frozen...', flush=True)
        probe = probe_check(source_features, source_ids, target_features, ids, folder)
        metrics['domain_separability'] = probe['accuracy']
        differences = {key: {'original': float(prior[name][key]), 'verified': value} for key, value in metrics.items() if abs(value - float(prior[name][key])) > 1e-10}
        check = {'selected_epoch': record['best_epoch'], 'checkpoint_sha256': sha(checkpoint_path),
                 'bn_matches_official_pretrained': bn_match, 'target_predictions_identical': target_same,
                 'metric_differences': differences, 'probe_convergence_warning': probe['convergence_warning'],
                 'probe_iterations': probe['iterations'], 'metrics': metrics}
        check['passed'] = (bn_match and target_same and not differences and not probe['convergence_warning'] and check['checkpoint_sha256'] == record['checkpoint_sha256'])
        summary[name] = check
        write_json(folder / 'verification.json', check)
        write_json(output / 'summary.json', summary)
        print(f"{name}: {'VERIFIED' if check['passed'] else 'NEEDS INSPECTION'}; probe iterations={probe['iterations']}", flush=True)
        if name in ('dann', 'cdan'):
            clearer_plot(read_csv(Path(record['directory']) / 'history.csv'), name, record['best_epoch'], folder / 'training_clearer.png')
        del model, saved, state, source_features, target_features
        torch.cuda.empty_cache()
    assert sha(lock_path) == FROZEN_SHA, 'Freeze file changed during verification.'
    write_json(output / 'files_manifest.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    archive = Path('/content') / (output.name + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(output))
    all_pass = all(v['passed'] for v in summary.values())
    print('ALL SIX VERIFIED' if all_pass else 'VERIFICATION NEEDS INSPECTION — do not change settings or overwrite original results.')
    print('New verification evidence saved:', output)
    files.download(str(archive))

if __name__ == '__main__':
    main()
