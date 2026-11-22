"""Double the orientation-preserving member weight while keeping verifier features and original rows fixed."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys

root = Path(os.environ['GO_ROOT']).resolve()
work = root / 'reproduction/cx38_candidate_pipeline1118'
day = root / 'reproduction/day_20261119'
config = json.loads((work / 'reproduction/review_candidate_config.json').read_text())
parent = work / 'sub/go19_cx38.csv'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == '0b34e38881dc40ba224d0e188f678d8d2f68b6bf3d76565ed3a083dfbe9ff5ca'
votes = config['weighted_members'] + ['cx38_selected'] * 3
environment = dict(os.environ, GO_ROOT=str(work), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')


def run(script, *arguments, extra=None):
    subprocess.run([sys.executable, str(work / 'src' / script), *map(str, arguments)], cwd=work,
                   env=dict(environment, **(extra or {})), check=True)


run('codex_ens_review_candidate_1118.py', 'cx38weight6', *votes, extra={'NEW_POOL_ONLY_MEMBERS': 'cx38_selected'})
run('codex_mk_gate_1118.py', parent, day / 'weight6_gate.csv', 2, 3, 1.4, 1., 'cx38weight6', 3, 1, .6, .5, .5,
    ','.join(config['original_members']), 0, extra={'SFINAL': 'cx38weight6'})
output = day / 'go19_cx38w6.csv'
run('codex_frozen_verifier_1118.py', parent, output, 'cx38weight6', ','.join(config['verifier_training_members']),
    ','.join(config['production_members']), 1., extra={'VERIFIER_MODEL_FILE': str(work / 'results/verifier_gate1118.txt')})
rows = list(csv.reader(output.open(newline='')))
control = list(csv.reader(parent.open(newline='')))
assert len(rows) == 1601 and [row[0] for row in rows] == [row[0] for row in control]
assert all(len(row) == 6 and len(set(row[1:])) == 5 for row in rows[1:])
assert all(a == b for a, b in zip(rows[1:], control[1:]) if a[0].startswith(('pub_q_', 'priv_q_')))
record = {'file': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
          'parent_score': .8621245, 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'hypothesis': 'Increase the publicly improving orientation-preserving member from weight three to six; freeze verifier member features.',
          'parameters': {'member': 'cx38_selected', 'weight': 6},
          'changed_top1': sum(a[1] != b[1] for a, b in zip(rows[1:], control[1:]))}
assert record['sha256'] != record['parent_sha256']
(day / 'weight6_candidate.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
