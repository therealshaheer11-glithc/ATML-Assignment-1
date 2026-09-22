"""Verify submitted small evidence without data downloads, inference, or training."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'task2'

def js(path):
    return json.loads(path.read_text())

def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def close(a, b):
    if abs(float(a) - float(b)) > 1e-10:
        raise ValueError(f'Metric mismatch: {a} versus {b}')

def main():
    for relative, expected in js(T / 'submission-manifest.json').items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    protocol = js(ROOT / 'shared/splits/pacs_sketch_seed6304.json')
    classes, sources = protocol['classes'], protocol['sources']
    seen = set()
    for domain in sources:
        partitions = protocol['source_splits'][domain]
        records = sorted(partitions['train'] + partitions['validation'], key=lambda r: r['path'])
        labels = [classes[r['class_id']] for r in records]
        train, val = train_test_split(np.arange(len(records)), test_size=.2, random_state=6304, stratify=labels)
        assert [records[int(i)] for i in sorted(train)] == partitions['train']
        assert [records[int(i)] for i in sorted(val)] == partitions['validation']
        for r in records:
            assert r['path'] not in seen and Path(r['path']).parent.name == classes[r['class_id']]
            seen.add(r['path'])
    images = rows(T / 'provenance/dataset_image_hashes.csv')
    assert len(images) == 9991 and len({r['sha256'] for r in images}) == 9991
    original = js(T / 'provenance/dan_strength_preregistration.json')
    assert original['recorded_utc'] == '2026-09-22T14:21:33.152051+00:00'
    lock = js(T / 'provenance/freeze.json')
    comparison = {r['run_id']: r for r in rows(T / 'results/final/comparison.csv')}
    predictions = rows(T / 'results/final/target_predictions.csv')
    source_predictions = rows(T / 'results/verification/source_predictions.csv')
    probe_predictions = rows(T / 'results/verification/probe_predictions.csv')
    probe_checks = js(T / 'results/verification/probe_checks.json')
    verification = js(T / 'results/verification/summary.json')
    id_truth = {r['id']: Path(r['path']).parent.name for r in protocol['target_unlabeled']}
    assert len(predictions) == len(id_truth) == 3929
    assert {r['id'] for r in predictions} == set(id_truth)
    assert all(r['true_class'] == id_truth[r['id']] for r in predictions)
    truth = [classes.index(r['true_class']) for r in predictions]
    for name, record in lock['runs'].items():
        directory = T / 'results/training' / name
        meta = js(directory / 'run.json')
        history = rows(directory / 'history.csv')
        assert [int(r['epoch']) for r in history] == list(range(1, len(history) + 1))
        best, selected, stale = float('-inf'), None, 0
        for epoch, row in enumerate(history, 1):
            assert stale < 5
            score = float(row['mean_source_macro_f1'])
            if score > best + 1e-12:
                best, selected, stale = score, epoch, 0
            else:
                stale += 1
        assert len(history) <= 30 and (len(history) == 30 or stale == 5)
        assert selected == meta['best_epoch'] == record['best_epoch'] == verification[name]['selected_epoch']
        assert meta['config'] == record['config'] and meta['target_labels_used'] is False
        assert meta['checkpoint_sha256'] == record['checkpoint_sha256'] == verification[name]['checkpoint_sha256']
        close(best, comparison[name]['mean_source_macro_f1'])
        selected_source = [r for r in source_predictions if r['run_id'] == name]
        for domain in sources:
            values = [r for r in selected_source if r['domain'] == domain]
            expected = {r['path']: r['class_id'] for r in protocol['source_splits'][domain]['validation']}
            assert len(values) == len(expected) and {r['path'] for r in values} == set(expected)
            assert all(int(r['truth']) == expected[r['path']] for r in values)
            y, yp = [int(r['truth']) for r in values], [int(r['prediction']) for r in values]
            close(accuracy_score(y, yp), comparison[name][domain + '_accuracy'])
            close(f1_score(y, yp, labels=list(range(7)), average='macro', zero_division=0), comparison[name][domain + '_macro_f1'])
        pred = [classes.index(r[name + '_prediction']) for r in predictions]
        acc = accuracy_score(truth, pred)
        close(acc, comparison[name]['target_accuracy'])
        close(f1_score(truth, pred, labels=list(range(7)), average='macro', zero_division=0), comparison[name]['target_macro_f1'])
        close(acc - float(comparison['source_only']['target_accuracy']), comparison[name]['target_accuracy_change_vs_source_only'])
        assert confusion_matrix(truth, pred, labels=list(range(7))).tolist() == js(T / 'results/final' / (name + '_confusion.json'))
        pp = [r for r in probe_predictions if r['run_id'] == name]
        assert len(pp) == 602 and len({r['index'] for r in pp}) == 602
        close(accuracy_score([int(r['truth']) for r in pp], [int(r['prediction']) for r in pp]), comparison[name]['domain_separability'])
        assert not probe_checks[name]['convergence_warning'] and max(probe_checks[name]['iterations']) < 2000
        assert verification[name]['passed'] and verification[name]['bn_matches_official_pretrained']
        print(f'{name}: saved evidence verified')
    with ZipFile(T / 'provenance/original-training-code.zip') as archive:
        for name in ['shared/pacs.py', 'shared/mmd.py', 'shared/make_pacs_protocol.py', 'task2/model.py', 'task2/methods.py', 'task2/train.py', 'task2/evaluate_final.py', 'task2/plot_results.py']:
            assert (ROOT / name).read_bytes() == archive.read(name), name
    print('PASS: six selected runs, exact source splits, source/target/probe metrics, original training code, and package integrity.')

if __name__ == '__main__':
    main()
