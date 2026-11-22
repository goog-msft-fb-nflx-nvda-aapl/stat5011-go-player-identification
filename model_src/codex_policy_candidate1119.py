"""Fuse the validated spatial-policy residual into the exact accepted pre-verifier scores."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

root = Path(os.environ['GO_ROOT']).resolve()
work = root / 'reproduction/policy_test1119'
parent_root = root / 'reproduction/cx38_candidate_pipeline1118'
for attempt in range(1200):
    if all((work / f'{pool}_summary.json').exists() for pool in ['pub2', 'priv2']):
        break
    time.sleep(30)
else:
    raise RuntimeError('Policy feature extraction did not complete in its ten-hour window')
parent = parent_root / 'sub/go19_nover.csv'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == 'abf65aebd37460b1279dc03177e5755660ff05063099bcae9f778ad21d5a1fa7'
rows = list(csv.reader(parent.open(newline='')))
replacement = {}
control = {}
coverage = {}


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def zrow(values):
    return (values - values.mean(1, keepdims=True)) / (values.std(1, keepdims=True) + 1e-9)


for pool in ['pub2', 'priv2']:
    data = np.load(work / f'{pool}_features.npz')
    features = data['features']
    valid = np.isfinite(features).all(1) & (data['counts'] > 0)
    reference_valid = valid[:6000].reshape(600, 10)
    fitted = features[:6000][valid[:6000]]
    mean = fitted.mean(0)
    scale = np.maximum(fitted.std(0), 1e-4)
    standardized = (np.where(valid[:, None], features, mean) - mean) / scale
    prototypes = normalize(standardized[:6000].reshape(600, 10, -1).sum(1) /
                           np.maximum(reference_valid.sum(1)[:, None], 1))
    queries = np.zeros((400, features.shape[1]), dtype=np.float32)
    query_valid = np.zeros(400, dtype=bool)
    query_indices = data['query_indices']
    for question in range(400):
        selected = (query_indices == question) & valid[6000:]
        if selected.any():
            queries[question] = standardized[6000:][selected].mean(0)
            query_valid[question] = True
    similarities = normalize(queries) @ prototypes.T
    candidate_valid = reference_valid.sum(1) > 0
    similarities[:, ~candidate_valid] = similarities[:, candidate_valid].mean(1, keepdims=True)
    similarities[~query_valid] = 0
    fingerprint = zrow(similarities)
    base = np.load(parent_root / 'results' / f'Sfinal_review_candidate_{pool}.npy')
    pids = data['player_ids']
    assert np.array_equal(pids, np.load(parent_root / 'results' / f'S_{pool}_pids.npy'))
    assert base.shape == fingerprint.shape == (400, 600) and np.isfinite(fingerprint).all()
    for question_id, base_order, order in zip(data['question_ids'], np.argsort(-base, axis=1)[:, :5],
                                             np.argsort(-(zrow(base) + .5 * fingerprint), axis=1)[:, :5]):
        control[str(question_id)] = [str(question_id)] + [str(pids[index]) for index in base_order]
        replacement[str(question_id)] = [str(question_id)] + [str(pids[index]) for index in order]
    coverage[pool] = {'reference_games_valid': int(reference_valid.sum()), 'queries_with_features': int(query_valid.sum()),
                      'candidates_with_features': int(candidate_valid.sum())}
    np.save(work / f'fingerprint_{pool}.npy', fingerprint)
control_rows = [rows[0]] + [control.get(row[0], row) for row in rows[1:]]
assert control_rows == rows, 'Accepted baseline control must be exact'
candidate = [rows[0]] + [replacement.get(row[0], row) for row in rows[1:]]
assert len(candidate) == 1601 and all(len(row) == 6 and len(set(row[1:])) == 5 for row in candidate[1:])
assert all(left == right for left, right in zip(candidate[1:], rows[1:]) if left[0].startswith(('pub_q_', 'priv_q_')))
output = work / 'go20_hslfp.csv'
with output.open('w', newline='') as handle:
    csv.writer(handle).writerows(candidate)
record = {'file': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'public_score': None,
          'parent_score': .8630397, 'weight': .5, 'feature': 'policy-and-trunk residual mean', 'profile': 'fixed rank_3d',
          'changed_top1': sum(left[1] != right[1] for left, right in zip(candidate[1:], rows[1:])), 'coverage': coverage,
          'handicap_policy': 'HD>=2 and failed board replays contribute a neutral feature score',
          'hypothesis': 'Frozen board-state policy residuals add player preference information beyond coordinate embeddings.'}
assert record['changed_top1'] > 0
(work / 'candidate_ready.json').write_text(json.dumps(record, indent=2) + '\n')
print('POLICY_CANDIDATE_READY', json.dumps(record), flush=True)
