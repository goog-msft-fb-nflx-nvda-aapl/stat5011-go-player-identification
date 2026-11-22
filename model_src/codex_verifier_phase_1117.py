"""Fixed phase policy evaluated alongside the existing hold-out verifier."""
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path('/home/jtan/solardefect/go_20261113')
os.environ.pop('NOEXTRA', None)
sys.argv = [__file__, 'a2,b1,b2,b3', 'a2,b1,b2,b3,h4,h5']
source = (root / 'src/ho_verifier2.py').read_text()
# Reuse the existing episode generator and exact verifier training configuration.
exec(compile(source.split('\nfor tags in te_sets:')[0], str(root / 'src/ho_verifier2.py'), 'exec'))
phases = {f'{tag}ph{phase}': np.load(root / f'results/HO_{tag}ph{phase}.npy', mmap_mode='r') for tag in ['b1', 'h4'] for phase in range(3)}
policy = {1: ([1], .5), 2: ([0, 1, 2], 1.), 3: ([1, 2], .25), 5: ([0, 1, 2], .5)}
results = {k: {} for k in policy}
for seed in range(200, 205):
    for k, (selected_phases, weight) in policy.items():
        episode_seed = seed * 10 + k
        tags = te_sets[0]
        features, labels, groups, base, truth, top = episode(B_, tags, episode_seed, k)
        scores = mdl.predict(features).reshape(len(truth), TOP)
        verifier = np.broadcast_to(scores.min(1)[:, None], base.shape).copy()
        for i in range(len(truth)):
            verifier[i, top[i]] = scores[i]
        blend = zs(verifier) + zs(base)
        rng = np.random.default_rng(episode_seed)
        pool = rng.choice(elig, 600, replace=False)
        games = np.array([rng.permutation(byp[p])[:10+k] for p in pool])
        phase_signal = 0
        for tag in ['b1', 'h4']:
            for phase in selected_phases:
                values = phases[f'{tag}ph{phase}']
                candidate = nrm(nrm(values[games[:, :10]].astype(np.float32)).mean(1))
                query = nrm(nrm(values[games[truth, 10:]].astype(np.float32)).mean(1))
                phase_signal = phase_signal + csls(asn(query @ candidate.T))
        phase_signal = zs(phase_signal / (2 * len(selected_phases)))
        variants = {'base': base, 'verifier': verifier, 'verifier_blend': blend,
                    'base_phase': zs(base) + weight * phase_signal,
                    'verifier_phase': zs(verifier) + weight * phase_signal,
                    'verifier_blend_phase': zs(blend) + weight * phase_signal}
        for name, values in variants.items():
            results[k].setdefault(name, []).append(score_rank(values, truth))
    print('evaluation seed', seed, 'complete', flush=True)
summary = {str(k): {name: float(np.concatenate(rows).mean()) for name, rows in group.items()} for k, group in results.items()}
means = {name: float(np.mean([summary[str(k)][name] for k in policy])) for name in results[1]}
report = {'per_game_count': summary, 'means': means, 'policy': policy, 'evaluation_seeds': list(range(200,205)), 'limitations': ['Previously used hold-out players, with fresh episode seeds.', 'Existing hold-out verifier with the production feature contract, not the complete verQ production ensemble and LM gate.', 'Original-family evaluation remains pending.'], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root / 'results/codex_verifier_phase_1117.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
