"""Copy and verify existing Task 2 evidence from Drive. No training or inference.

Run in Colab: python export_task2_evidence.py [--notebook PATH]
The generated ZIP is for review before importing its task2/ tree into GitHub.
"""
import argparse
import csv
import hashlib
import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RUNS = ('source_only', 'dan_0p1', 'dan_1', 'dan_10', 'dann', 'cdan')
CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
FREEZE_SHA = 'baacc896c12285216eee120b785475b69c1e3bfcee911c33a759a72c897f15b5'
AUDIT_SHA = '4567df8de5d2b86d662c8c38017df1d0004b626b461c793d767207422033c1ca'
ALLOWED = {'.json', '.csv', '.md', '.txt', '.png', '.ipynb'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-root', type=Path, default=Path('/content/drive/MyDrive/ATML-PA1'))
    parser.add_argument('--output-dir', type=Path, default=Path('/content'))
    parser.add_argument('--notebook', type=Path)
    args = parser.parse_args()
    base = args.drive_root.resolve()
    require(base.is_dir(), f'Drive root missing: {base}')
    freeze_path = base / 'task2_official_freeze_20260923/task2_checkpoint_freeze.json'
    audit_path = base / 'task2_official_freeze_20260923/source_checkpoint_audit.json'
    final = base / 'task2_final_evaluation_20260923'
    require(sha(freeze_path) == FREEZE_SHA, 'Freeze hash mismatch; stop and preserve files.')
    require(sha(audit_path) == AUDIT_SHA, 'Source audit hash mismatch.')
    freeze, audit = read(freeze_path), read(audit_path)
    require(tuple(freeze['official_runs']) == RUNS, 'Unexpected frozen runs or order.')
    require(freeze['source_audit_sha256'] == AUDIT_SHA, 'Freeze source-audit identity mismatch.')
    require(freeze.get('target_labels_accessed') is False and audit.get('target_labels_accessed') is False, 'Pre-target flags failed.')
    require(freeze.get('status') == 'FROZEN_BEFORE_TARGET_LABEL_ACCESS', 'Invalid freeze status.')
    require(audit.get('status') == 'SOURCE_CHECKPOINT_AUDIT_PASS', 'Invalid source audit status.')
    completion = read(final / 'evaluation_complete.json')
    require(completion.get('status') == 'FINAL_EVALUATION_COMPLETE', 'Evaluation incomplete.')
    require(completion.get('freeze_sha256') == FREEZE_SHA, 'Evaluation freeze mismatch.')
    require(completion.get('target_labels_used_for_training_or_selection') is False, 'Evaluation training/selection flag failed.')
    require(completion.get('target_count') == 3929, 'Unexpected target count.')
    manifest = read(final / 'EVALUATION-MANIFEST.json')
    require(manifest.get('status') == 'COMPLETE', 'Evaluation manifest incomplete.')
    require(len(manifest['files']) == 25 and len({x['path'] for x in manifest['files']}) == 25, 'Expected 25 unique manifested result files.')
    for item in manifest['files']:
        path = (final / item['path']).resolve()
        require(path.is_relative_to(final.resolve()), 'Unsafe manifest path.')
        require(path.is_file() and sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], f'Evaluation artifact differs: {item["path"]}')
    predictions = rows(final / 'target_predictions.csv')
    require(len(predictions) == 3929 and len({r['id'] for r in predictions}) == 3929, 'Target IDs/count invalid.')
    results = read(final / 'final_results.json')['runs']
    require(set(results) == set(RUNS), 'Unexpected final result runs.')
    for name in RUNS:
        require(all(r['true_class'] in CLASSES and r[name + '_prediction'] in CLASSES for r in predictions), f'Invalid class name: {name}')
        correct = sum(r['true_class'] == r[name + '_prediction'] for r in predictions)
        scores = []
        for label in CLASSES:
            tp = sum(r['true_class'] == label == r[name + '_prediction'] for r in predictions)
            count = sum(r['true_class'] == label for r in predictions) + sum(r[name + '_prediction'] == label for r in predictions)
            scores.append(2 * tp / count if count else 0)
        for metric, value in [('accuracy', correct / 3929), ('macro_f1', sum(scores) / 7)]:
            require(math.isclose(value, results[name]['target'][metric], abs_tol=1e-12), f'{name} target {metric} mismatch.')
    record = {'created_utc': datetime.now(timezone.utc).isoformat(), 'training_performed': False,
              'checkpoint_files_modified': False, 'freeze_sha256': FREEZE_SHA,
              'copied_files': [], 'missing_optional_evidence': []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / ('ATML-PA1-Task2-final-evidence-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.zip')
    # Stage only in a temporary local directory. Existing Drive artifacts are read-only inputs.
    with tempfile.TemporaryDirectory(prefix='atml_evidence_') as temporary:
        stage = Path(temporary)

        def copy(source, destination):
            require(source.is_file() and not source.is_symlink(), f'Expected regular file: {source}')
            require(source.suffix.lower() in ALLOWED, f'Unexpected export type: {source}')
            require(source.stat().st_size <= 25 * 1024 * 1024, f'Oversized evidence requires review: {source}')
            target = stage / destination
            require(not target.exists(), f'Duplicate destination: {destination}')
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            target.write_bytes(data)
            require(sha(source) == hashlib.sha256(data).hexdigest(), f'Source changed while exporting: {source}')
            record['copied_files'].append({'source': str(source), 'path': str(destination), 'bytes': len(data), 'sha256': sha(target)})

        for path in sorted(final.rglob('*')):
            if path.is_file():
                copy(path, Path('task2/results/final') / path.relative_to(final))
        for path in sorted(freeze_path.parent.iterdir()):
            if path.is_file() and path.suffix in {'.json', '.md', '.txt', '.csv'}:
                copy(path, Path('task2/provenance/freeze') / path.name)
        for name, frozen in freeze['official_runs'].items():
            checkpoint = Path(frozen['checkpoint_path']).resolve()
            require(checkpoint.is_relative_to(base), f'Checkpoint outside expected Drive root: {name}')
            require(sha(checkpoint) == frozen['checkpoint_sha256'], f'Frozen checkpoint differs: {name}')
            require(results[name]['checkpoint_sha256'] == frozen['checkpoint_sha256'], f'Evaluation checkpoint differs: {name}')
            history = rows(checkpoint.parent / 'history.csv')
            run = read(checkpoint.parent / 'run.json')
            best = max(history, key=lambda r: float(r['mean_source_macro_f1']))
            require(int(best['epoch']) == int(frozen['selected_epoch']) == int(run['best_epoch']), f'Selection mismatch: {name}')
            require(len(history) == run['epochs_completed'], f'History count mismatch: {name}')
            require([int(r['epoch']) for r in history] == list(range(1, len(history) + 1)), f'Nonconsecutive history: {name}')
            require(math.isclose(float(best['mean_source_macro_f1']), run['best_mean_source_macro_f1'], abs_tol=1e-12), f'Source F1/history mismatch: {name}')
            require(math.isclose(run['best_mean_source_macro_f1'], results[name]['mean_source_validation_macro_f1'], abs_tol=1e-8), f'Source F1/evaluation mismatch: {name}')
            require(run['best_checkpoint_sha256'] == frozen['checkpoint_sha256'] and run.get('target_labels_used') is False, f'Run identity mismatch: {name}')
            for filename in ('history.csv', 'run.json', 'best_source_validation.json'):
                copy(checkpoint.parent / filename, Path('task2/results/training') / name / filename)
        versions = {
            'v1': 'task2_corrected_20260923',
            'v2': 'task2_corrected_clipped_v2_20260923',
            'v3': 'task2_corrected_normalized_v3_20260923',
            'v4': 'task2_adversarial_normalized_v4_20260923',
        }
        pilot_runs = {'v1': ('source_only', 'dan_0p1', 'dan_1'), 'v2': ('dan_1',), 'v3': ('dann',), 'v4': ()}
        for version, directory in versions.items():
            source = base / directory
            if not source.is_dir():
                record['missing_optional_evidence'].append(str(source)); continue
            # Version-level setup, locks, preregistration, adoption and recovery records.
            for path in sorted(source.iterdir()):
                if path.is_file() and path.suffix in {'.json', '.md', '.txt'}:
                    copy(path, Path('task2/provenance/versions') / version / path.name)
            for subdir in ('preregistration', 'initialization', 'diagnostics'):
                for path in sorted((source / subdir).rglob('*')):
                    if path.is_file() and path.suffix in {'.json', '.md', '.txt', '.csv'}:
                        copy(path, Path('task2/provenance/versions') / version / path.relative_to(source))
            pilot_root = source / "runs" if version == "v1" else source
            for name in pilot_runs[version]:
                for filename in ('history.csv', 'run.json', 'best_source_validation.json'):
                    path = pilot_root / name / filename
                    if path.is_file():
                        copy(path, Path('task2/provenance/pilots') / version / name / filename)
                    else:
                        record['missing_optional_evidence'].append(str(path))
        adoption = base / versions['v4'] / 'V4_ADOPTION_DECISION.json'
        require(adoption.is_file(), f'Missing adoption record: {adoption}')
        if args.notebook:
            require(args.notebook.suffix.lower() == '.ipynb', 'Supply the current Colab .ipynb notebook.')
            copy(args.notebook.resolve(), Path('task2/provenance/notebooks/corrected-task2-execution.ipynb'))
        else:
            record['missing_optional_evidence'].append('Latest corrected Colab notebook: download separately; needed for setup/recovery/adoption/freeze cells.')
        manifest_path = stage / 'task2/provenance/FINAL-EVIDENCE-EXPORT.json'
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(record, indent=2) + '\n')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    output.write(path, path.relative_to(stage).as_posix())
    print('VERIFIED: final artifact hashes, six checkpoint hashes, saved source selections, 3,929 target predictions per run and target metrics.')
    print('EXPORT:', archive)
    print('COPIED FILES:', len(record['copied_files']))
    print('MISSING OPTIONAL EVIDENCE:', json.dumps(record['missing_optional_evidence'], indent=2))
    print('Training performed: False; existing files changed: False. Review ZIP before publication.')


if __name__ == '__main__':
    main()
