"""Audit published Task 1 tables and pinned Task 2 packages; no training/inference."""
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T1 = ROOT / 'task1/results'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, context):
    assert math.isclose(float(actual), float(expected), abs_tol=2e-7), (context, actual, expected)


def macro_f1(data, truth, prediction, classes=10):
    scores = []
    for label in range(classes):
        tp = sum(int(r[truth]) == label == int(r[prediction]) for r in data)
        count = sum(int(r[truth]) == label for r in data) + sum(int(r[prediction]) == label for r in data)
        scores.append(2 * tp / count if count else 0)
    return statistics.mean(scores)


def task1():
    manifests = {name: rows(T1 / 'splits' / filename) for name, filename in [
        ('train', 'head_train_indices.csv'), ('validation', 'validation_indices.csv'),
        ('test', 'evaluation_subset.csv')]}
    for name, count, per_class in [('train', 4000, 400), ('validation', 1000, 100), ('test', 500, 50)]:
        data = manifests[name]
        assert len(data) == count and len({r['official_index'] for r in data}) == count
        assert Counter(r['class_id'] for r in data) == {str(k): per_class for k in range(10)}
        assert {r['official_partition'] for r in data} == ({'test'} if name == 'test' else {'train'})
    assert not ({r['official_index'] for r in manifests['train']} & {r['official_index'] for r in manifests['validation']})
    eval_truth = {r['official_index']: r['class_id'] for r in manifests['test']}
    clean = defaultdict(list)
    for r in rows(T1 / 'clean/clean_predictions.csv'):
        clean[r['predictor']].append(r)
    clean_metrics = read(T1 / 'clean/clean_metrics.json')['results']
    assert set(clean) == set(clean_metrics) and len(clean) == 4
    for name, data in clean.items():
        assert len(data) == 500 and {r['official_index']: r['true_class_id'] for r in data} == eval_truth
        m = clean_metrics[name]
        n = sum(r['true_class_id'] == r['predicted_class_id'] for r in data)
        assert n == m['correct_count']
        close(n / 500, m['top1_accuracy'], name)
        close(macro_f1(data, 'true_class_id', 'predicted_class_id'), m['macro_f1'], name)
        close(statistics.mean(float(r['maximum_confidence']) for r in data), m['mean_maximum_confidence'], name)
    lookup = {name: {r['official_index']: r['predicted_class_id'] for r in data} for name, data in clean.items()}

    def transformed(data, metrics, predictor, pred_key):
        assert len(data) == 500 and {r['official_index']: r['true_class_id'] for r in data} == eval_truth
        assert all(r['clean_predicted_class_id'] == lookup[predictor][r['official_index']] for r in data)
        correct = sum(r['true_class_id'] == r[pred_key] for r in data)
        consistent = sum(r['clean_predicted_class_id'] == r[pred_key] for r in data)
        assert correct == metrics['correct_count'] and consistent == metrics['prediction_match_count']
        close(correct / 500, metrics['absolute_accuracy'], predictor)
        close(consistent / 500, metrics['prediction_consistency_with_clean'], predictor)
        close(correct / 500 - clean_metrics[predictor]['top1_accuracy'], metrics['accuracy_change_from_own_clean_baseline'], predictor)

    color = defaultdict(list)
    for r in rows(T1 / 'color/color_predictions.csv'):
        color[(r['predictor'], r['condition'])].append(r)
    assert len(color) == 8
    for (name, condition), data in color.items():
        transformed(data, read(T1 / 'color/color_metrics.json')['results'][name][condition], name, 'transformed_predicted_class_id')
    for name in clean:
        data = rows(T1 / 'patch_shuffle' / f'{name}_predictions.csv')
        transformed(data, read(T1 / 'patch_shuffle' / f'{name}.json'), name, 'shuffled_predicted_class_id')
        for r in data:
            permutation = json.loads(r['permutation_0_based_json'])
            assert sorted(permutation) == list(range(16)) and permutation != list(range(16))
        for d in (8, 16, 32):
            m = read(T1 / 'translation' / f'{name}__d{d}_summary.json')
            for direction in ('up', 'down', 'left', 'right'):
                data = rows(T1 / 'translation' / f'{name}__d{d}_{direction}_predictions.csv')
                transformed(data, m['direction_results'][direction], name, 'transformed_predicted_class_id')
            close(statistics.mean(v['absolute_accuracy'] for v in m['direction_results'].values()), m['mean_accuracy'], name)
            close(statistics.mean(v['prediction_consistency_with_clean'] for v in m['direction_results'].values()), m['mean_prediction_consistency_with_clean'], name)

    selection = read(T1 / 'cue_conflict/selection_record.json')
    for filename, key in [('accepted_conflicts.csv', 'accepted_conflicts_sha256'), ('final_review_manifest.csv', 'final_review_manifest_sha256')]:
        assert sha(T1 / 'cue_conflict' / filename) == selection[key]
    accepted = rows(T1 / 'cue_conflict/accepted_conflicts.csv')
    assert len(accepted) == 200
    assert len(rows(T1 / 'cue_conflict/final_review_manifest.csv')) == 265
    accepted_ids = {r['candidate_id'] for r in accepted}
    for name in clean:
        data = rows(T1 / 'cue_conflict' / f'{name}_predictions.csv')
        assert len(data) == 200 and {r['candidate_id'] for r in data} == accepted_ids
        m = read(T1 / 'cue_conflict' / f'{name}.json')
        counts = Counter('shape' if r['predicted_class_id'] == r['shape_class_id'] else 'texture' if r['predicted_class_id'] == r['texture_class_id'] else 'other' for r in data)
        for decision in ('shape', 'texture', 'other'):
            assert counts[decision] == m[decision + '_decision_count']
        close(100 * counts['shape'] / (counts['shape'] + counts['texture']), m['shape_bias_percent'], name)
        close(100 * (counts['shape'] + counts['texture']) / 200, m['coverage_percent'], name)
        groups = defaultdict(list)
        for r in rows(T1 / 'feature_similarity' / f'{name}_pairs.csv'):
            groups[r['condition']].append(r)
        summaries = read(T1 / 'feature_similarity' / f'{name}.json')['results']
        for condition, m in summaries.items():
            data = groups[condition] if 'mean_four_directions' not in condition else sum((groups[f'{condition.split("_")[0]}_{direction}'] for direction in ('up','down','left','right')), [])
            assert len(data) == m['number_of_pairs']
            close(statistics.mean(float(r['cosine_similarity']) for r in data), m['mean_cosine_similarity'], condition)
            close(statistics.median(float(r['cosine_similarity']) for r in data), m['median_cosine_similarity'], condition)

    for name in ('resnet50', 'vit_b_16', 'clip_vit_b_32'):
        data = rows(T1 / 'training' / f'{name}_history.csv')
        m = read(T1 / 'training' / f'{name}_summary.json')['result']
        best = max(data, key=lambda r: float(r['validation_accuracy']))
        assert len(data) == m['epochs_completed'] and int(best['epoch']) == m['best_epoch']
        close(best['validation_accuracy'], m['best_validation_accuracy'], name)
        record = read(T1 / 'representation/umap' / f'{name}_record.json')
        folder = T1 / 'representation/umap'
        assert sha(folder / f'{name}_coordinates.csv') == record['coordinate_csv_sha256']
        assert sha(folder / f'{name}_comparison.png') == record['figure_png_sha256']
        assert sha(T1 / 'representation/umap_selection.csv') == record['selection_manifest_sha256']
        coords = rows(folder / f'{name}_coordinates.csv')
        assert len(coords) == 1000 and set(Counter(r['condition'] for r in coords).values()) == {200}
    compact = read(T1 / 'comparison/compact_comparison_record.json')
    assert sha(T1 / 'comparison/compact_comparison.csv') == compact['csv_sha256']
    assert sha(T1 / 'comparison/compact_comparison.png') == compact['figure_sha256']
    print('PASS: Task 1 splits, head selections, clean/color/patch/translation/cue metrics, cosine summaries, UMAP and comparison hashes.')


def task2_packages():
    for name, commit in [('PACKAGE-MANIFEST.json', 'd26997b22d3b7722e2ecc828dde1445244afc04b'), ('FINAL-EVALUATION-MANIFEST.json', 'edecd5b9429799cc51c9e96625191beaf45562af')]:
        m = read(ROOT / 'task2' / name)
        for item in m['files']:
            data = subprocess.check_output(['git', 'show', f'{commit}:{item["path"]}'], cwd=ROOT)
            assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
            assert len(data) == item['bytes']
        print(f'PASS: {name}, {len(m["files"])} files at pinned commit {commit[:12]}.')
    assert sha(ROOT / 'shared/splits/pacs_sketch_seed6304.json') == 'e0f075e1e4f2c43c7db2423bb9b31f901d4e1157e72c2097b3e156501ce2dc74'
    print('PENDING: Task 2 actual Drive evidence export and final artifact verification; no neural checkpoint inference performed here.')


def links():
    for path in ROOT.rglob('*.md'):
        if '.git' in path.parts:
            continue
        for href in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in href and not href.startswith('#'):
                assert (path.parent / href.split('#')[0]).exists(), (path.relative_to(ROOT), href)
    print('PASS: local Markdown link targets.')


if __name__ == '__main__':
    task1()
    task2_packages()
    links()
