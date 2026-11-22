"""Does absolute board orientation identify Go players? (CPU only, minutes.)

Every neural ID model is trained with random 8-fold symmetry augmentation and scored with 4-symmetry TTA, which makes
its embedding (nearly) orientation-invariant. If players habitually open in the same corner / with the same
orientation of their first moves, that information is discarded. This script measures it on the 800 hold-out players
(same split as train_idft.py: default_rng(1234), 800 players, >=15 games).

Per game it extracts, for the TARGET player's first three non-pass moves:
  orient : which of the 8 symmetries maps the move triple to its canonical (lexicographically smallest) form
           (class 8 = symmetric/ambiguous); this is pure orientation, invisible to an orientation-invariant model
  shape  : the canonical triple itself (orientation-free opening shape)
Then a smoothed per-player multinomial ("naive Bayes") is fitted on 10 reference games and evaluated on 1/2/3/5 query
games among 600 candidates with the competition metric. With --ho_tags it also fuses the orientation log-likelihood
into the cached hold-out embeddings (results/HO_<tag>.npy) to measure the INCREMENTAL gain without any training.

usage: python -I orientation_signal.py --root $GO_ROOT [--ho_tags a2,b1,b2,b3,h4,h5] [--seeds 5]
"""
import argparse, json, math, os
import numpy as np

W5 = np.exp(-np.arange(5))
SYMS = [lambda x, y: (x, y), lambda x, y: (18 - x, y), lambda x, y: (x, 18 - y), lambda x, y: (18 - x, 18 - y),
        lambda x, y: (y, x), lambda x, y: (18 - y, x), lambda x, y: (y, 18 - x), lambda x, y: (18 - y, 18 - x)]


def first_own(M, C, col, k=3, chunk=100000):
    out = np.full((len(M), k), 361, np.int64)
    for s in range(0, len(M), chunk):
        m = M[s:s + chunk].astype(np.int64); own = (C[s:s + chunk] == col[s:s + chunk, None]) & (m < 361)
        idx = np.argsort(~own, axis=1, kind='stable')[:, :k]
        val = np.take_along_axis(m, idx, 1); ok = np.take_along_axis(own, idx, 1)
        out[s:s + chunk] = np.where(ok, val, 361)
    return out


def canon(F):
    codes = []
    for f in SYMS:
        cols = []
        for j in range(F.shape[1]):
            p = F[:, j]; x, y = p // 19, p % 19; X, Y = f(x, y)
            cols.append(np.where(p < 361, X * 19 + Y, 361))
        c = np.zeros(len(F), np.int64)
        for v in cols:
            c = c * 362 + v
        codes.append(c)
    codes = np.stack(codes, 1); mn = codes.min(1)
    hits = codes == mn[:, None]; orient = np.where(hits.sum(1) > 1, 8, hits.argmax(1))
    return orient, mn


def metric(S, truth):
    o = np.argsort(-S, 1)[:, :5]
    return ((o == truth[:, None]) * W5[None]).sum(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=os.environ.get('GO_ROOT'))
    ap.add_argument('--ho_tags', default=''); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--alpha', type=float, default=2.)
    ap.add_argument('--fusion_feature', choices=['orient_x_colour', 'colour', 'orient_given_colour'], default='orient_x_colour')
    a = ap.parse_args(); assert a.root
    z = np.load(f'{a.root}/data/train_arr.npz'); M, C, col, P = z['M'], z['C'], z['col'], z['P']
    M = np.where(M == 65535, 361, M)                                    # treat padding like a pass (never an own move)
    F = first_own(M, C, col); orient, shape = canon(F)
    hold = np.random.default_rng(1234).choice(np.unique(P), 800, replace=False); ishold = np.isin(P, hold)
    rep = {'orient_population_counts_B': np.bincount(orient[(~ishold) & (col == 0)], minlength=9).round(0).tolist(),
           'orient_population_counts_W': np.bincount(orient[(~ishold) & (col == 1)], minlength=9).round(0).tolist()}
    # player consistency: modal orientation share per player and colour (players with >=20 games of that colour)
    for c_ in (0, 1):
        sel = (~ishold) & (col == c_) & (orient < 8); pp = P[sel]; oo = orient[sel]
        tab = np.zeros((P.max() + 1, 9)); np.add.at(tab, (pp, oo), 1); n = tab.sum(1); keep = n >= 20
        rep[f'modal_share_per_player_colour{c_}'] = float((tab[keep].max(1) / n[keep]).mean())
        pop = tab.sum(0); rep[f'modal_share_population_colour{c_}'] = float(pop.max() / pop.sum())
    feats = {'orient': orient, 'shape': shape, 'orient_x_colour': orient * 2 + col, 'colour': col}
    pop = {}
    for name, f in feats.items():
        u, inv = np.unique(f, return_inverse=True); feats[name] = inv
        if name == 'orient_x_colour':
            joint_colour = u % 2
        cnt = np.bincount(inv[~ishold], minlength=len(u)) + .5; pop[name] = cnt / cnt.sum()
    feats['orient_given_colour'] = feats['orient_x_colour']
    pop['orient_given_colour'] = pop['orient_x_colour']
    rep['fusion_feature'] = a.fusion_feature
    meta = None; HO = {}
    if a.ho_tags:
        meta = np.load(f'{a.root}/results/HO_meta.npz'); gidx = meta['idx']
        for t in a.ho_tags.split(','):
            e = np.load(f'{a.root}/results/HO_{t}.npy').astype(np.float32); HO[t] = e / np.linalg.norm(e, axis=1, keepdims=True)
        rowof = {g: i for i, g in enumerate(gidx)}
    cnt = np.bincount(P, minlength=P.max() + 1); elig = np.array([p for p in hold if cnt[p] >= 15])
    if meta is not None:
        available = np.isin(np.arange(len(P)), gidx)
        byp = {p: np.flatnonzero((P == p) & available) for p in elig}
        elig = np.array([p for p in elig if len(byp[p]) >= 15])
    else:
        byp = {p: np.flatnonzero(P == p) for p in elig}
    res = {}
    for seed in range(a.seeds):
        r = np.random.default_rng(seed); pl = r.choice(elig, 600, replace=False)
        refs = []; qry = []
        for p in pl:
            g = r.permutation(byp[p]); refs.append(g[:10]); qry.append(g[10:15])
        refs = np.array(refs); qry = np.array(qry)
        LL = {}
        for name, f in feats.items():
            V = len(pop[name]); tab = np.zeros((600, V)); np.add.at(tab, (np.repeat(np.arange(600), 10), f[refs].ravel()), 1)
            prob = (tab + a.alpha * pop[name][None]) / (10 + a.alpha)   # Dirichlet smoothing toward the population
            LL[name] = np.log(prob) - np.log(pop[name])[None]              # (600 players, V) log-likelihood ratio vs population
        LL['orient_given_colour'] = LL['orient_x_colour'] - LL['colour'][:, joint_colour]
        for k in (1, 2, 3, 5):
            for name in feats:
                S = sum(LL[name][:, feats[name][qry[:, j]]].T for j in range(k))     # (600 queries, 600 players)
                res.setdefault(f'{name}_alone/k{k}', []).append(metric(S, np.arange(600)).mean())
                if k == 1:
                    bits = LL[name][np.arange(600), feats[name][qry[:, 0]]] / math.log(2)
                    res.setdefault(f'{name}_bits_per_game', []).append(bits.mean())
            if HO:
                Z = 0
                for t, E in HO.items():
                    Pc = E[[[rowof[g] for g in row] for row in refs]].mean(1); Pc /= np.linalg.norm(Pc, axis=1, keepdims=True)
                    Q = E[[[rowof[g] for g in row[:k]] for row in qry]].mean(1); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
                    S = Q @ Pc.T; Z = Z + (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
                Z = Z / len(HO)
                for w in (0, .1, .2, .35, .5):
                    S = Z + w * sum(LL[a.fusion_feature][:, feats[a.fusion_feature][qry[:, j]]].T for j in range(k))
                    res.setdefault(f'HO_ens+{w}*orient/k{k}', []).append(metric(S, np.arange(600)).mean())
    rep['results'] = {k: round(float(np.mean(v)), 4) for k, v in sorted(res.items())}
    rep['random_baseline_metric'] = round(float(W5.sum() / 600), 4)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
