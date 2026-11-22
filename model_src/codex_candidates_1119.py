"""Build controlled orientation and original-family fusion experiments from the scored cx38 parent."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_orientation_signal_1118 import first_own, canon
from gocommon import parse

root = Path(os.environ['GO_ROOT']).resolve()
work = root / 'reproduction/cx38_candidate_pipeline1118'
output = root / 'reproduction/day_20261119'
output.mkdir(exist_ok=True)
csv.field_size_limit(10**9)
parent = work / 'sub/go19_cx38.csv'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == '0b34e38881dc40ba224d0e188f678d8d2f68b6bf3d76565ed3a083dfbe9ff5ca'
assert hashlib.sha256((work / 'sub/cx38_export_replay.csv').read_bytes()).hexdigest() == hashlib.sha256(parent.read_bytes()).hexdigest()
rows = list(csv.reader(parent.open(newline='')))
config = json.loads((work / 'reproduction/review_candidate_config.json').read_text())
results = work / 'results'
tests = work / 'data/tests1v2/tests'


def zrow(values):
    return (values - values.mean(1, keepdims=True)) / (values.std(1, keepdims=True) + 1e-9)


def softmax(values, temperature):
    values = values / (temperature[:, None] if np.ndim(temperature) else temperature)
    values = values - values.max(1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(1, keepdims=True)


def orientation(records):
    first = np.full((len(records), 3), 361, dtype=np.int64)
    colours = []
    for index, (sgf, target) in enumerate(records):
        target = 0 if target == 'B' else 1
        colours.append(target)
        moves = [(x * 19 + y) for colour, x, y in parse(sgf)[1] if colour == target and x >= 0][:3]
        first[index, :len(moves)] = moves
    oriented, _ = canon(first)
    return oriented * 2 + np.array(colours), np.array(colours)


records = []
def write_candidate(name, replacement, hypothesis, parameters):
    path = output / name
    candidate = [rows[0]] + [replacement.get(row[0], row) for row in rows[1:]]
    assert len(candidate) == 1601
    assert [row[0] for row in candidate] == [row[0] for row in rows]
    assert all(len(row) == 6 and len(set(row[1:])) == 5 for row in candidate[1:])
    assert candidate != rows, 'Candidate duplicates its parent'
    with path.open('w', newline='') as handle:
        csv.writer(handle).writerows(candidate)
    record = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'parent_score': 0.8621245, 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
              'hypothesis': hypothesis, 'parameters': parameters,
              'changed_top1': sum(left[1] != right[1] for left, right in zip(candidate[1:], rows[1:]))}
    records.append(record)


z = np.load(root / 'data/train_arr.npz')
first = first_own(z['M'], z['C'], z['col'])
orient, _ = canon(first)
players = z['P']
hold = np.random.default_rng(1234).choice(np.unique(players), 800, replace=False)
training = ~np.isin(players, hold)
joint_counts = np.bincount((orient * 2 + z['col'])[training], minlength=18).astype(float) + .5
colour_counts = np.bincount(z['col'][training], minlength=2).astype(float) + .5
population_joint = joint_counts / joint_counts.sum()
population_colour = colour_counts / colour_counts.sum()
(output / 'orientation_population.json').write_text(json.dumps({'joint': population_joint.tolist(), 'colour': population_colour.tolist()}, indent=2))
replacements = {'joint': {}, 'conditional': {}}
for pool, suffix in [('pub2', 'public'), ('priv2', 'private')]:
    candidates = list(csv.DictReader((tests / f'player_identification_candidates2_{suffix}.csv').open()))
    by_player = {}
    for row in candidates:
        by_player.setdefault(row['player_id'], []).append(row)
    player_ids = sorted(by_player)
    assert np.array_equal(np.load(results / f'S_{pool}_pids.npy'), player_ids)
    ordered = [row for player in player_ids for row in by_player[player]]
    joint, colour = orientation([(row['sgf_content'], row['color']) for row in ordered])
    joint_table = np.zeros((600, 18))
    colour_table = np.zeros((600, 2))
    np.add.at(joint_table, (np.repeat(np.arange(600), 10), joint), 1)
    np.add.at(colour_table, (np.repeat(np.arange(600), 10), colour), 1)
    joint_ll = np.log((joint_table + 2 * population_joint[None]) / 12) - np.log(population_joint)[None]
    colour_ll = np.log((colour_table + 2 * population_colour[None]) / 12) - np.log(population_colour)[None]
    questions = list(csv.DictReader((tests / f'player_identification_test2_{suffix}.csv').open()))
    final = np.load(results / 'final_export1119' / f'final_{pool}.npy')
    for index, row in enumerate(questions):
        query = [(row[f'sgf_{game}'], row[f'color_{game}']) for game in range(1, int(row['num_games']) + 1)]
        query_joint, query_colour = orientation(query)
        joint_score = joint_ll[:, query_joint].sum(1)
        conditional_score = joint_score - colour_ll[:, query_colour].sum(1)
        for feature, score in [('joint', joint_score), ('conditional', conditional_score)]:
            order = np.argsort(-(final[index] + .2 * score))[:5]
            replacements[feature][row['question_id']] = [row['question_id']] + [player_ids[candidate] for candidate in order]
write_candidate('go19_orient.csv', replacements['joint'], 'Add explicit absolute-orientation and colour likelihood to cx38 final scores.', {'weight': .2, 'alpha': 2, 'feature': 'orientation_x_colour'})
write_candidate('go19_ocond.csv', replacements['conditional'], 'Isolate orientation conditional on target colour from the joint orientation-colour signal.', {'weight': .2, 'alpha': 2, 'feature': 'orientation_given_colour'})

for weight, gate_weight, filename in [(3, 3, 'go19_oldw3.csv'), (3, 0, 'go19_oldnogate.csv')]:
    replacement = {}
    for pool, suffix in [('pub', 'public'), ('priv', 'private')]:
        questions = list(csv.DictReader((tests / f'player_identification_test_{suffix}.csv').open()))
        player_ids = np.load(results / f'S_{pool}_pids.npy')
        base = zrow(np.load(results / f'S_{pool}.npy'))
        members = [zrow(np.load(results / f'Sidft_{tag}_{pool}.npy')) for tag in config['original_weighted_members']]
        identity = zrow(np.mean(members, axis=0).astype(np.float32))
        counts = np.array([int(row['num_games']) for row in questions], float)
        candidate_rank = np.load(results / f'LPc_{pool}.npy')
        query_rank = np.load(results / f'LPq_{pool}.npy')
        gate = zrow(np.log(softmax(query_rank, np.maximum(1., counts / 5)) @ softmax(candidate_rank, 30).T + 1e-3))
        for current_weight, current_gate in [(1, 3), (weight, gate_weight)]:
            scores = (base + current_weight * identity + current_gate * gate) / 1.4
            scores -= scores.max()
            for iteration in range(30):
                scores -= np.logaddexp.reduce(scores, axis=1, keepdims=True)
                scores -= np.logaddexp.reduce(scores, axis=0, keepdims=True)
            rebuilt = {row['question_id']: [row['question_id']] + [str(player_ids[c]) for c in order]
                       for row, order in zip(questions, np.argsort(-scores, axis=1)[:, :5])}
            if current_weight == 1:
                assert all(row == rebuilt[row[0]] for row in rows[1:] if row[0].startswith(pool + '_q_')), 'Original-family control failed exact replay'
            else:
                replacement.update(rebuilt)
    write_candidate(filename, replacement, 'Test original-family fusion and rank-gate dependence while keeping all new-family rows at the scored cx38 parent.', {'original_identity_weight': weight, 'rank_gate_weight': gate_weight, 'sinkhorn_temperature': 1.4, 'iterations': 30})
assert len({record['sha256'] for record in records}) == len(records)
(output / 'candidates.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records, indent=2), flush=True)
