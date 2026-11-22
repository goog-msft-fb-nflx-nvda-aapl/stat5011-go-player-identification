"""Build a new-member candidate from the verified 0.8630397 pre-verifier parent."""
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
parser.add_argument('--mode', choices=['inference', 'artifacts'], default='inference')
a = parser.parse_args()
root = a.root.resolve()
config = json.loads((root / 'reproduction/new_member_config.json').read_text())
parent = root / 'sub/accepted_parent.csv'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == 'abf65aebd37460b1279dc03177e5755660ff05063099bcae9f778ad21d5a1fa7'
environment = dict(os.environ, GO_ROOT=str(root), PYTHONHASHSEED='0', NEW_FAMILY_ONLY='1')


def run(script, *arguments, extra=None):
    subprocess.run([sys.executable, str(root / 'src' / script), *map(str, arguments)],
                   cwd=root, env=dict(environment, **(extra or {})), check=True)


def fuse(name, votes, output):
    run('codex_ens_review_candidate_1118.py', name, *votes)
    run('codex_mk_gate_1118.py', parent, output, 2, 3, 1.4, 1., name, 3, 1, .6, .5, .5,
        ','.join(config['original_members']), 0, extra={'SFINAL': name})


control = root / 'sub/control_replay.csv'
fuse('newmember_control', config['weighted_members'], control)
assert hashlib.sha256(control.read_bytes()).hexdigest() == hashlib.sha256(parent.read_bytes()).hexdigest()
tag = config['new_member']
if a.mode == 'inference':
    run('codex_infer_ablation_1118.py', '--root', root, '--tag', tag)
output = root / 'sub' / config['output_name']
fuse('newmember_candidate', config['weighted_members'] + [tag] * config['new_member_weight'], output)
rows = list(csv.reader(output.open(newline='')))
baseline = list(csv.reader(parent.open(newline='')))
assert len(rows) == 1601 and [row[0] for row in rows] == [row[0] for row in baseline]
assert all(len(row) == 6 and len(set(row[1:])) == 5 for row in rows[1:])
assert all(left == right for left, right in zip(rows[1:], baseline[1:]) if left[0].startswith(('pub_q_', 'priv_q_')))
record = {'verified': True, 'public_score': None, 'file': str(output),
          'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'parent_score': .8630397,
          'parent_control_exact': True, 'config': config,
          'changed_top1': sum(left[1] != right[1] for left, right in zip(rows[1:], baseline[1:]))}
(root / 'results/candidate_ready.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
