"""Encoder-only original-family pilot with disjoint 80-reference and 5/10/15/20-query games."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--seeds', type=int, default=10)
a = parser.parse_args()
base_tags = ['a2', 'b1', 'b1', 'b2', 'b3', 'c25', 'c25', 'c25', 'c26', 'c26', 'c26']
groups = {'original_six': base_tags,
          'plus_c33_c34': base_tags + ['c33'] * 3 + ['c34'] * 3,
          'plus_orientation': base_tags + ['c33'] * 3 + ['c34'] * 3 + ['cx38_selected'] * 3}


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def zrow(values):
    return (values - values.mean(1, keepdims=True)) / (values.std(1, keepdims=True) + 1e-9)


def metric(scores):
    order = np.argsort(-scores, axis=1)[:, :5]
    return ((order == np.arange(400)[:, None]) * np.exp(-np.arange(5))[None]).sum(1)


report = {'limitations': ['Representative encoder-only pilot, six baseline models plus three additions.',
                           'Five-view features and rank gates are omitted.',
                           'Unique-answer pseudo-episodes mirror the existing original-family assignment assumption.',
                           'Candidate players are reused across seeds. Query games are excluded from reference games.'],
          'groups': groups, 'pools': {}}
for pool in ['pub', 'priv']:
    embeddings = {tag: normalize(np.load(a.root / 'results' / f'G_{tag}_{pool}_cand.npy').astype(np.float32))
                  for tag in sorted(set(groups['plus_orientation']))}
    assert all(values.shape == (400, 100, 256) for values in embeddings.values())
    collected = {}
    for seed in range(a.seeds):
        rng = np.random.default_rng(191900 + seed)
        indices = np.stack([rng.permutation(100) for _ in range(400)])
        ks = rng.permutation(np.repeat([5, 10, 15, 20], 100))
        scores = {}
        for tag, values in embeddings.items():
            reference = normalize(np.stack([values[player, indices[player, :80]].mean(0) for player in range(400)]))
            query = normalize(np.stack([values[player, indices[player, 80:80 + ks[player]]].mean(0) for player in range(400)]))
            scores[tag] = zrow(query @ reference.T)
        for group, tags in groups.items():
            average = np.mean([scores[tag] for tag in tags], axis=0)
            for temperature in [0, .7, 1.4, 2.0]:
                output = average.copy()
                if temperature:
                    output /= temperature
                    for iteration in range(30):
                        output -= np.logaddexp.reduce(output, axis=1, keepdims=True)
                        output -= np.logaddexp.reduce(output, axis=0, keepdims=True)
                values = metric(output)
                key = f'{group}/sinkhorn_{temperature}'
                collected.setdefault(key, []).append(float(values.mean()))
    baseline = np.array(collected['original_six/sinkhorn_1.4'])
    summary = {}
    for key, values in collected.items():
        values = np.array(values)
        paired = values - baseline
        summary[key] = {'mean': float(values.mean()), 'paired_gain': float(paired.mean()),
                        'paired_standard_error': float(paired.std(ddof=1) / np.sqrt(a.seeds)),
                        'seed_scores': values.tolist()}
    report['pools'][pool] = summary
    print(pool, json.dumps(summary), flush=True)
(a.root / 'results/original_selfsim_summary.json').write_text(json.dumps(report, indent=2) + '\n')
print('ORIGINAL_FAMILY_PILOT_COMPLETE', flush=True)
