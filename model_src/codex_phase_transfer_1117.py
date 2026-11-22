"""CPU-only phase late-fusion selection on A players and evaluation on B players."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/jtan/solardefect/go_20261113')
TAGS = ['a2', 'b1', 'b2', 'b3', 'h4', 'h5']
KS = [1, 2, 3, 5]
meta = np.load(ROOT / 'results/HO_meta.npz')
players = meta['P']
by_player = {p: np.flatnonzero(players == p) for p in np.unique(players)}
eligible = sorted(p for p, rows in by_player.items() if len(rows) >= 15)
permutation = np.random.default_rng(0).permutation(eligible)
sides = {'development': set(permutation[:358]), 'evaluation': set(permutation[358:716])}
paths = {t: ROOT / f'results/HO_{t}.npy' for t in TAGS}
paths.update({f'{t}ph{phase}': ROOT / f'results/HO_{t}ph{phase}.npy' for t in ['b1', 'h4'] for phase in range(3)})
embeddings = {key: np.load(path, mmap_mode='r') for key, path in paths.items()}

def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-9)

def row_z(values):
    return (values - values.mean(1, keepdims=True)) / (values.std(1, keepdims=True) + 1e-9)

def score_rows(values, truth):
    order = np.argsort(-values, axis=1)[:, :5]
    return ((order == truth[:, None]) * np.exp(-np.arange(5))[None]).sum(1)

def similarity(key, candidate_rows, query_rows):
    values = embeddings[key]
    candidate = normalize(normalize(values[candidate_rows].astype(np.float32)).mean(1))
    query = normalize(normalize(values[query_rows].astype(np.float32)).mean(1))
    return query @ candidate.T

def calibrated(values):
    row_top = np.sort(values, axis=1)[:, -20:]
    col_top = np.sort(values, axis=0)[-20:]
    values = row_z(.5 * ((values - row_top.mean(1)[:, None]) / (row_top.std(1)[:, None] + 1e-9) + (values - col_top.mean(0)[None]) / (col_top.std(0)[None] + 1e-9)))
    return values - .9 * np.sort(values, axis=0)[-3:].mean(0)[None]

results = {}
for split, seeds in [('development', range(20, 23)), ('evaluation', range(100, 105))]:
    records = {k: {} for k in KS}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pool = rng.choice(eligible, 600, replace=False)
        games = [rng.permutation(by_player[p])[:15] for p in pool]
        games = np.array(games)
        query_indices = np.array([i for i, p in enumerate(pool) if p in sides[split]])
        candidate_rows = games[:, :10]
        for k in KS:
            query_rows = games[query_indices, 10:10+k]
            baseline = row_z(sum(calibrated(similarity(t, candidate_rows, query_rows)) for t in TAGS))
            signals = {}
            phase_scores = [sum(calibrated(similarity(f'{t}ph{phase}', candidate_rows, query_rows)) for t in ['b1', 'h4']) / 2 for phase in range(3)]
            signals['all_phases'] = row_z(sum(phase_scores) / 3)
            signals['exclude_opening'] = row_z((phase_scores[1] + phase_scores[2]) / 2)
            signals['middle'] = row_z(phase_scores[1])
            signals['end'] = row_z(phase_scores[2])
            candidates = {'baseline': baseline}
            for name, signal in signals.items():
                for weight in [.25, .5, 1.0]:
                    candidates[f'{name}_w{weight}'] = baseline + weight * signal
            for name, values in candidates.items():
                records[k].setdefault(name, []).append(score_rows(values, query_indices))
        print(split, 'seed', seed, 'complete', flush=True)
    results[split] = {str(k): {name: {'mean': float(np.concatenate(rows).mean()), 'count': int(sum(len(row) for row in rows))} for name, rows in group.items()} for k, group in records.items()}

selected = {str(k): max(results['development'][str(k)], key=lambda name: results['development'][str(k)][name]['mean']) for k in KS}
comparison = []
for k in KS:
    key = str(k)
    baseline = results['evaluation'][key]['baseline']['mean']
    score = results['evaluation'][key][selected[key]]['mean']
    comparison.append({'query_games': k, 'development_selected': selected[key], 'baseline': baseline, 'selected_score': score, 'gain': score-baseline})
report = {'description': __doc__, 'selection': selected, 'evaluation': comparison, 'mean_gain': float(np.mean([row['gain'] for row in comparison])), 'results': results, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'limitations': ['Cached hold-out reused in earlier experiments, so this is exploratory validation.', 'Tests the new-family pool-600 setting only.', 'No leaderboard upload or full-production verifier comparison.']}
output = ROOT / 'results/codex_phase_transfer_1117.json'
output.write_text(json.dumps(report, indent=2))
print(json.dumps({'evaluation': comparison, 'mean_gain': report['mean_gain'], 'report': str(output)}, indent=2), flush=True)
