"""CPU-only checks for the NEW-family Task 1 pools, using per-game embeddings already on disk.

Inputs (written by src/idft_games.py for every production member):
  $ROOT/results/G_<tag>_<pool>_cand.npy  (600, 10, 256) candidate game embeddings, candidates in sorted-pid order
  $ROOT/results/G_<tag>_<pool>_q.npy     (n_query_games, 256)
  $ROOT/results/G_<tag>_<pool>_qidx.npy  (n_query_games,) question index of each query game

Three sub-commands:
  selfsim  Labelled self-simulation on the test candidates themselves (production all-player models can be
           scored here, unlike on the 800 reused hold-out players). Pseudo-queries are k of a candidate's own
           10 games; the remaining games stay as that candidate's references. Compares raw z-averaged cosine,
           CSLS, AS-norm+CSLS, and capacity-1 assignment, under unique and budget-constrained repeated-player sampling.
The repeated regime is a finite-game proxy and differs from independent uniform sampling.
  repeat   Estimates how often two REAL test questions come from the same player, by counting query pairs whose
           query-query cosine exceeds thresholds calibrated on same-player / different-player splits of
           candidate games. Compare the count with the two hypotheses printed next to it.
  overlap  Tests whether pub2 and priv2 candidate lists contain the same physical players (half-prototype
           matching calibrated on within-candidate 5/5 splits).

usage:
  python -I selfsim_new.py selfsim --root $GO_ROOT --tags a2,b1,...,cx35_lm6 [--seeds 8] [--pools pub2,priv2]
  python -I selfsim_new.py repeat  --root $GO_ROOT --tags ...
  python -I selfsim_new.py overlap --root $GO_ROOT --tags ...
Exact numbers depend on the member list; use the production member list for decisions.
"""
import argparse, json, math, os
import numpy as np
from scipy.optimize import linear_sum_assignment

W5 = np.exp(-np.arange(5))


def nrm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def zrow(A):
    return (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)


def metric(S, truth):
    o = np.argsort(-S, 1)[:, :5]
    return ((o == truth[:, None]) * W5[None]).sum(1)


def load(root, tags, pool):
    Ec, Eq, qi = [], [], None
    for t in tags:
        Ec.append(nrm(np.load(f'{root}/results/G_{t}_{pool}_cand.npy').astype(np.float32)))
        Eq.append(nrm(np.load(f'{root}/results/G_{t}_{pool}_q.npy').astype(np.float32)))
        q = np.load(f'{root}/results/G_{t}_{pool}_qidx.npy')
        assert qi is None or (qi == q).all(), 'qidx differs between members'
        qi = q
    return np.stack(Ec), np.stack(Eq), qi          # (M,600,10,256), (M,nq_games,256), (nq_games,)


def ens_score(Ec, ref_mask, qsets):
    """Ec (M,C,10,d); ref_mask (C,10) bool; qsets list of (cand, game_idx array). -> z-averaged cosine (nq, C)."""
    w = ref_mask[None, :, :, None].astype(np.float32)
    P = nrm((Ec * w).sum(2) / np.maximum(w.sum(2), 1))                       # (M,C,d)
    Q = nrm(np.stack([Ec[:, c, g].mean(1) for c, g in qsets], 1))              # (M,nq,d)
    return np.mean([zrow(Q[m] @ P[m].T) for m in range(len(Ec))], 0)


def csls(Z, a=.6, k=3):
    return Z - a * np.sort(Z, 0)[-k:].mean(0)[None]


def asn(S, n=20):
    sq = np.sort(S, 1)[:, -n:]; sc = np.sort(S, 0)[-n:]
    return zrow(.5 * ((S - sq.mean(1)[:, None]) / (sq.std(1)[:, None] + 1e-9) + (S - sc.mean(0)[None]) / (sc.std(0)[None] + 1e-9)))


def cap_sinkhorn(Z, T=.5, iters=300):
    """Each candidate answers at most ~one question: extra dummy row absorbs the unused candidates."""
    nq, nc = Z.shape
    if nc <= nq:
        return Z
    A = np.vstack([Z / T, np.zeros((1, nc))])
    rs = np.r_[np.zeros(nq), np.log(nc - nq)]
    for _ in range(iters):
        A = A - (np.logaddexp.reduce(A, 1, keepdims=True) - rs[:, None])
        A = A - np.logaddexp.reduce(A, 0, keepdims=True)
    return A[:nq]


def hungarian(Z):
    r, c = linear_sum_assignment(-Z); S = Z.copy(); S[r, c] += 100; return S


def episode(rng, C, mode, sym=True, budget=5):
    """Returns ref_mask (C,10), qsets, truth, ks. 100 queries of each k in 1/2/3/5, as in the test."""
    ks = rng.permutation(np.repeat([1, 2, 3, 5], 100))
    ref = np.ones((C, 10), bool); used = np.zeros(C, int); qsets = []; truth = []
    perm = [rng.permutation(10) for _ in range(C)]
    order = rng.permutation(C)
    for i, k in enumerate(ks):
        if mode == 'unique':
            c = int(order[i])
        else:
            ok = np.flatnonzero(used + k <= budget); c = int(rng.choice(ok))
        g = perm[c][used[c]:used[c] + k]; used[c] += k; ref[c, g] = False
        qsets.append((c, g)); truth.append(c)
    if sym:   # drop the same NUMBER of references from untouched candidates, so true and distractor prototypes are comparable
        drops = rng.choice(used[used > 0], size=int((used == 0).sum()))
        for c, d in zip(np.flatnonzero(used == 0), drops):
            ref[c, perm[c][:d]] = False
    return ref, qsets, np.array(truth), ks


def cmd_selfsim(a):
    out = {}
    for pool in a.pools:
        Ec, _, _ = load(a.root, a.tags, pool); C = Ec.shape[1]
        for mode in ('unique', 'repeat'):
            res = {}
            for seed in range(a.seeds):
                rng = np.random.default_rng(1000 + seed)
                ref, qsets, truth, ks = episode(rng, C, mode)
                Z = ens_score(Ec, ref, qsets)
                variants = {'raw': Z, 'csls.6': csls(Z, .6), 'csls.9': csls(Z, .9), 'asn+csls.9': csls(asn(Z), .9),
                            'cap_sinkhorn_T.5': cap_sinkhorn(Z, .5), 'cap_sinkhorn_T1': cap_sinkhorn(Z, 1.), 'hungarian': hungarian(Z)}
                for name, S in variants.items():
                    m = metric(S, truth)
                    res.setdefault(name, {}).setdefault('all', []).append(m.mean())
                    for k in (1, 2, 3, 5):
                        res[name].setdefault(str(k), []).append(m[ks == k].mean())
            summ = {n: {k: [round(float(np.mean(v)), 4), round(float(np.std(v) / math.sqrt(len(v))), 4)] for k, v in d.items()} for n, d in res.items()}
            out[f'{pool}/{mode}'] = summ
            print(pool, mode, json.dumps({n: d['all'] for n, d in summ.items()}), flush=True)
    print(json.dumps(out, indent=1))


def pair_cos(Ec, rng, ka, kb, n_same=4000, n_diff=200000):
    """same-player cosine of disjoint ka/kb game subsets of one candidate vs different-candidate subsets (ensemble mean)."""
    M, C = Ec.shape[:2]
    def proto(c, g):
        return nrm(Ec[:, c, g].mean(1))                      # (M,d)
    same = []
    for _ in range(n_same):
        c = rng.integers(C); g = rng.permutation(10)
        same.append(np.mean((proto(c, g[:ka]) * proto(c, g[ka:ka + kb])).sum(-1)))
    # different players: precompute random subset prototypes
    PA = np.stack([proto(c, rng.permutation(10)[:ka]) for c in range(C)], 1)   # (M,C,d)
    PB = np.stack([proto(c, rng.permutation(10)[:kb]) for c in range(C)], 1)
    D = np.mean([PA[m] @ PB[m].T for m in range(M)], 0); np.fill_diagonal(D, np.nan)
    diff = D[~np.isnan(D)]
    return np.array(same), diff


def cmd_repeat(a):
    rng = np.random.default_rng(7)
    for pool in a.pools:
        Ec, Eq, qi = load(a.root, a.tags, pool); C = Ec.shape[1]
        nq = qi.max() + 1; k = np.bincount(qi, minlength=nq)
        Qm = np.stack([nrm(np.stack([Eq[m, qi == j].mean(0) for j in range(nq)])) for m in range(len(Eq))])   # (M,nq,d)
        A = np.mean([Qm[m] @ Qm[m].T for m in range(len(Qm))], 0)
        print(f'== {pool}: questions {nq}, k histogram {dict(zip(*np.unique(k, return_counts=True)))}')
        for ka, kb in ((5, 5), (5, 3), (3, 3)):
            idx_a = np.flatnonzero(k == ka); idx_b = np.flatnonzero(k == kb)
            if ka == kb:
                iu = np.triu_indices(len(idx_a), 1); real = A[np.ix_(idx_a, idx_a)][iu]; npairs = len(real)
            else:
                real = A[np.ix_(idx_a, idx_b)].ravel(); npairs = len(real)
            same, diff = pair_cos(Ec, rng, ka, kb)
            exp_rep = npairs / C          # expected same-player pairs if questions pick players uniformly with replacement
            print(f'  pair type {ka}v{kb}: {npairs} real pairs; same-player pairs expected: unique=0, with-replacement={exp_rep:.1f}')
            for tpr in (.5, .7, .9):
                tau = np.quantile(same, 1 - tpr); fpr = (diff > tau).mean()
                obs = int((real > tau).sum())
                print(f'    tau={tau:.3f} (same-player recall {tpr:.0%}, diff-player FPR {fpr:.2e}): observed {obs} | '
                      f'expected if unique {fpr * npairs:.1f} | if with-replacement {fpr * npairs + tpr * exp_rep:.1f}')


def cmd_overlap(a):
    rng = np.random.default_rng(11)
    E1, _, _ = load(a.root, a.tags, a.pools[0]); E2, _, _ = load(a.root, a.tags, a.pools[1])
    same, diff = pair_cos(E1, rng, 5, 5)
    h1 = nrm(E1[:, :, :5].mean(2)); h2 = nrm(E2[:, :, 5:].mean(2))
    X = np.mean([h1[m] @ h2[m].T for m in range(len(h1))], 0)
    best = X.max(1); arg = X.argmax(1); mutual = X.argmax(0)[arg] == np.arange(len(arg))
    print(f'{a.pools[0]} vs {a.pools[1]}: 5-game half-prototypes; {X.size} cross pairs')
    for tpr in (.5, .7, .9):
        tau = np.quantile(same, 1 - tpr); fpr = (diff > tau).mean()
        print(f'  tau={tau:.3f} recall {tpr:.0%} FPR {fpr:.2e}: mutual-NN pairs above tau {int((mutual & (best > tau)).sum())} '
              f'(expected false {fpr * X.size:.1f}; full overlap would give ~{tpr * len(best):.0f})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['selfsim', 'repeat', 'overlap'])
    ap.add_argument('--root', default=os.environ.get('GO_ROOT'))
    ap.add_argument('--tags', required=True)
    ap.add_argument('--pools', default='pub2,priv2')
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args(); assert a.root, 'set --root or GO_ROOT'
    a.tags = a.tags.split(','); a.pools = a.pools.split(',')
    {'selfsim': cmd_selfsim, 'repeat': cmd_repeat, 'overlap': cmd_overlap}[a.cmd](a)
