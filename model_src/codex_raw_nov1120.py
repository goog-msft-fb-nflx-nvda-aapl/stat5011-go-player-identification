"""Prepare canonical raw-data variants to test the legacy base-view reconstruction discrepancy."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
a = parser.parse_args()
root = a.root.resolve()
source = root / 'reproduction/full_inference_cx35_portable1118'
work = root / 'reproduction/raw_nov1120'
work.mkdir(exist_ok=False)
for folder in ['src', 'results', 'sub', 'data/tests1v2', 'reproduction']:
    (work / folder).mkdir(parents=True, exist_ok=True)
for path in (source / 'src').glob('*.py'):
    shutil.copy2(path, work / 'src' / path.name)
for name in ['codex_mk_gate_1118.py', 'codex_ens_review_candidate_1118.py', 'codex_infer_ablation_1118.py']:
    shutil.copy2(root / 'reproduction/experiments_1119/src' / name, work / 'src' / name)
for path in (source / 'results').glob('*.npy'):
    assert path.resolve().is_relative_to(source), 'Raw stage must contain generated arrays, not production score links'
    (work / 'results' / path.name).symlink_to(path)
selected = root / 'reproduction/cx38_candidate_pipeline1118/results/idft_cx38_selected.pt'
assert hashlib.sha256(selected.read_bytes()).hexdigest() == '8614b9c681ab8fb5986f7de79224c6ff7dc37f805800c9b7147438d6702d8e08'
(work / 'results/idft_cx38_selected.pt').symlink_to(selected)
(work / 'results/lm_lm4.pt').symlink_to(source / 'results/lm_lm4.pt')
(work / 'data/tests1v2/tests').symlink_to(root / 'data/tests1v2/tests', target_is_directory=True)
template = work / 'sub/raw_original_template.csv'
shutil.copy2(source / 'sub/original_family_1117.csv', template)
config = json.loads((source / 'reproduction/task1_cx35_1118_config.json').read_text())
config['expected_submission_sha256'] = '538f7494d88869e2c1e198b73f8f607b643da4df58397b43a838ba6deffdc61c'
config['public_score'] = None
(work / 'reproduction/raw_parent_config.json').write_text(json.dumps(config, indent=2))
environment = dict(os.environ, GO_ROOT=str(work), PYTHONHASHSEED='0', NEW_FAMILY_ONLY='1',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')


def run(script, *arguments, extra=None):
    subprocess.run([sys.executable, str(work / 'src' / script), *map(str, arguments)], cwd=work,
                   env=dict(environment, **(extra or {})), check=True)


run('codex_infer_ablation_1118.py', '--root', work, '--tag', 'cx38_selected')
run('codex_ens_review_candidate_1118.py', 'rawnov1120', *config['weighted_members'], *(['cx38_selected'] * 3))
run('codex_mk_gate_1118.py', template, work / 'sub/go20_raw.csv', 2, 3, 1.4, 1., 'rawnov1120', 3, 1, .6, .5, .5,
    ','.join(config['original_members']), 0, extra={'SFINAL': 'rawnov1120'})
parent = root / 'reproduction/cx38_candidate_pipeline1118/sub/go19_nover.csv'
parent_rows = list(csv.reader(parent.open(newline='')))
raw_rows = list(csv.reader((work / 'sub/go20_raw.csv').open(newline='')))
assert len(raw_rows) == 1601 and [row[0] for row in raw_rows] == [row[0] for row in parent_rows]
assert all(len(row) == 6 and len(set(row[1:])) == 5 for row in raw_rows[1:])
replacement = {}
for pool in ['pub2', 'priv2']:
    base = np.load(work / 'results' / f'Sfinal_rawnov1120_{pool}.npy')
    fingerprint = np.load(root / 'reproduction/policy_test1119' / f'fingerprint_{pool}.npy')
    pids = np.load(work / 'results' / f'S_{pool}_pids.npy')
    qids = np.load(root / 'reproduction/policy_test1119' / f'{pool}_features.npz')['question_ids']
    standardized = (base - base.mean(1, keepdims=True)) / (base.std(1, keepdims=True) + 1e-9)
    for question, order in zip(qids, np.argsort(-(standardized + .5 * fingerprint), axis=1)[:, :5]):
        replacement[str(question)] = [str(question)] + [str(pids[index]) for index in order]
combined = [raw_rows[0]] + [replacement.get(row[0], row) for row in raw_rows[1:]]
with (work / 'sub/go20_rawfp.csv').open('w', newline='') as handle:
    csv.writer(handle).writerows(combined)
records = []
for filename in ['go20_raw.csv', 'go20_rawfp.csv']:
    path = work / 'sub' / filename
    rows = list(csv.reader(path.open(newline='')))
    records.append({'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'public_score': None,
                    'parent_score': .8630397, 'raw_base_view_seed': 0,
                    'changed_rows': sum(left != right for left, right in zip(rows[1:], parent_rows[1:])),
                    'changed_top1': sum(left[1] != right[1] for left, right in zip(rows[1:], parent_rows[1:])),
                    'hypothesis': 'Canonical raw base views' + (' plus spatial policy features' if 'rawfp' in filename else '')})
(work / 'candidate_ready.json').write_text(json.dumps(records, indent=2))
stage = root / 'reproduction/auto_backup_1118/raw_nov1120'
stage.mkdir(exist_ok=False)
archive = stage / 'raw_nov1120.tar.gz'
files = [work / 'candidate_ready.json', work / 'reproduction/raw_parent_config.json', Path(__file__)]
files += list((work / 'sub').glob('*.csv')) + list((work / 'src').glob('*.py'))
files += list((work / 'results').glob('Sfinal_rawnov1120_*.npy'))
with tarfile.open(archive, 'w:gz', dereference=True) as handle:
    for path in files:
        handle.add(path, arcname=str(path.relative_to(root)))
manifest = {'release_tag': 'go-raw-reproduction-20261120',
            'source_commit': (root / 'reproduction/raw_nov_source_commit1120.txt').read_text().strip(),
            'assets': [{'path': str(archive), 'name': archive.name, 'bytes': archive.stat().st_size,
                        'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}]}
(stage / 'backup_manifest.json').write_text(json.dumps(manifest, indent=2))
print('RAW_VARIANTS_READY_BACKUP_QUEUED', json.dumps(records), flush=True)
