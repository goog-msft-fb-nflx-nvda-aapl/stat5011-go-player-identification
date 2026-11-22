"""Build a checkpoint-matched experimental candidate while preserving accepted original-family rows."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
config = json.loads((root / 'reproduction/review_candidate_config.json').read_text())
parent = root / 'sub/go18_cx35.csv'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == config['parent_sha256']
environment = dict(os.environ, GO_ROOT=str(root), PYTHONHASHSEED='0')


def run(script, *arguments, extra=None):
    subprocess.run([sys.executable, str(root / 'src' / script), *map(str, arguments)],
                   cwd=root, env=dict(environment, **(extra or {})), check=True)


tag = config['new_member']
run('codex_infer_ablation_1118.py', '--root', root, '--tag', tag)
run('codex_ens_review_candidate_1118.py', 'review_candidate', *config['weighted_members'],
    extra={'NEW_POOL_ONLY_MEMBERS': tag})
run('codex_mk_gate_1118.py', parent, root / 'sub/review_gate.csv', 2, 3, 1.4, 1.,
    'review_candidate', 3, 1, .6, .5, .5, ','.join(config['original_members']), 0,
    extra={'SFINAL': 'review_candidate'})
output = root / 'sub/go19_cx38.csv'
run('codex_frozen_verifier_1118.py', parent, output, 'review_candidate',
    ','.join(config['verifier_training_members']), ','.join(config['production_members']), 1.,
    extra={'VERIFIER_MODEL_FILE': str(root / 'results/verifier_gate1118.txt')})
rows = list(csv.reader(output.open(newline='')))
parent_rows = list(csv.reader(parent.open(newline='')))
assert len(rows) == 1601 and [row[0] for row in rows] == [row[0] for row in parent_rows]
assert all(len(row) == 6 and len(set(row[1:])) == 5 for row in rows[1:])
assert all(row == baseline for row, baseline in zip(rows[1:], parent_rows[1:])
           if row[0].startswith(('pub_q_', 'priv_q_')))
record = {'state': 'inference_verified', 'public_score': None, 'file': str(output),
          'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
          'changed_top1': sum(row[1] != baseline[1] for row, baseline in zip(rows[1:], parent_rows[1:])),
          'parent_score': 0.8605620, 'config': config}
(root / 'results/review_candidate_ready.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
