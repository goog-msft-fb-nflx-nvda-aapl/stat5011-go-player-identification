"""KataGo HumanSL (b18c384nbt-humanv0) per-move features, run in PyTorch on GPU.
For every game and every move i<MAXM: log pi_r(played move) for R rank profiles, plus value/lead of the position (profile 'mid').
usage: python hsl_extract.py <src: train|test2|testid_X> <out.npy> [--n N] [--maxm 160] [--nproc 24]
Output float16 array (G, MAXM, R+3): [logp_r..., win(player to move), lead, mover colour(0/1, -1 pad)]  (NaN pad for logp)
"""
import sys, time, argparse, numpy as np, torch, csv, multiprocessing as mp
KG = (__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/katago/kgsrc/python')
sys.path.insert(0, KG); sys.path.insert(0, (__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
csv.field_size_limit(10**9)
PROFILES = ['20k', '10k', '5k', '2k', '1d', '3d', '5d', '8d']
INV = {'20k': 29, '10k': 19, '5k': 14, '2k': 11, '1d': 9, '3d': 7, '5d': 5, '8d': 2}
D = (__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/')
STAR = {2: [(15, 3), (3, 15)], 3: [(15, 3), (3, 15), (15, 15)], 4: [(15, 3), (3, 15), (15, 15), (3, 3)]}
STAR[5] = STAR[4] + [(9, 9)]; STAR[6] = STAR[4] + [(3, 9), (15, 9)]; STAR[7] = STAR[6] + [(9, 9)]
STAR[8] = STAR[6] + [(9, 3), (9, 15)]; STAR[9] = STAR[8] + [(9, 9)]

_state = {}
def _init():
    from katago.game.gamestate import GameState
    from katago.game.features import Features
    from katago.game.board import Board
    from katago.train import modelconfigs
    _state.update(GS=GameState, Board=Board)

def worker(a):
    """a = (moves[(c,x,y)], hd, maxm, cfgdict) -> (bin uint8 (n,C,19,19), glob float16 (n,G), tgt int16 (n), mover int8 (n))"""
    moves, hd, maxm, cfg, start = a
    GS, Board = _state['GS'], _state['Board']
    from katago.game.features import Features
    feats = _state.get('feats') or Features(cfg, 19); _state['feats'] = feats
    gs = GS(19, GS.RULES_CHINESE)
    try:
        if hd >= 2:
            for (x, y) in STAR[min(hd, 9)]: gs.board.set_stone(Board.BLACK, gs.board.loc(x, y))
            gs.boards = [gs.board.copy()]
        bins, globs, tgts, movers = [], [], [], []
        for j, (c, x, y) in enumerate(moves[:maxm]):
            pla = Board.BLACK if c == 0 else Board.WHITE
            gs.board.set_pla(pla)
            if j < start:
                loc0 = Board.PASS_LOC if x < 0 else gs.board.loc(x, y); gs.play(pla, loc0); continue
            b, g = gs.get_input_features(feats)
            loc = Board.PASS_LOC if x < 0 else gs.board.loc(x, y)
            tp = 361 if x < 0 else feats.loc_to_tensor_pos(loc, gs.board)
            bins.append(b[0].astype(np.uint8)); globs.append(g[0].astype(np.float16)); tgts.append(tp); movers.append(c)
            gs.play(pla, loc)
    except Exception as e:
        pass
    if not bins: return None
    return np.stack(bins), np.stack(globs), np.array(tgts, np.int16), np.array(movers, np.int8)

def games_from_src(src, n):
    if src == 'train':
        z = np.load(D + 'data/train_arr.npz'); M, C, hd = z['M'], z['C'], z['hd']; idx = np.random.default_rng(0).permutation(len(M))[:n]
        out = []
        for g in idx:
            m = M[g]; k = int(np.argmax(m == 65535)) if (m == 65535).any() else len(m)
            out.append(([(int(C[g, j]), -1 if m[j] == 361 else int(m[j]) // 19, -1 if m[j] == 361 else int(m[j]) % 19) for j in range(k)], int(hd[g])))
        return out, idx

    if src == 'trainsel':
        z = np.load(D + 'data/train_arr.npz'); M, C, hd, P = z['M'], z['C'], z['hd'], z['P']
        players = np.unique(P); hold = np.random.default_rng(1234).choice(players, 800, replace=False); rest = np.setdiff1d(players, hold)
        r = np.random.default_rng(7); trp = r.choice(rest, 2400, replace=False); order = np.argsort(P, kind='stable'); starts = {}
        sp = P[order]; bounds = np.searchsorted(sp, np.arange(P.max() + 2))
        idx = []
        for p in trp: g = order[bounds[p]:bounds[p + 1]]; idx += list(r.choice(g, min(25, len(g)), replace=False))
        for p in hold: g = order[bounds[p]:bounds[p + 1]]; idx += list(r.choice(g, min(40, len(g)), replace=False))
        idx = np.array(idx); out = []
        for g in idx:
            m = M[g]; k = int(np.argmax(m == 65535)) if (m == 65535).any() else len(m)
            out.append(([(int(C[g, j]), -1 if m[j] == 361 else int(m[j]) // 19, -1 if m[j] == 361 else int(m[j]) % 19) for j in range(k)], int(hd[g])))
        return out, idx
    if src == 'trainsel3':
        z = np.load(D + 'data/train_arr.npz'); M, C, hd, P = z['M'], z['C'], z['hd'], z['P']
        players = np.unique(P); hold = np.random.default_rng(1234).choice(players, 800, replace=False); rest = np.setdiff1d(players, hold)
        r = np.random.default_rng(7); trp = r.choice(rest, 2400, replace=False); rest2 = np.setdiff1d(rest, trp); r2 = np.random.default_rng(11); trp2 = r2.choice(rest2, min(3000, len(rest2)), replace=False); rest3 = np.setdiff1d(rest2, trp2); r3 = np.random.default_rng(13)
        order = np.argsort(P, kind='stable'); sp = P[order]; bounds = np.searchsorted(sp, np.arange(P.max() + 2)); idx = []
        for p in rest3: g = order[bounds[p]:bounds[p + 1]]; idx += list(r3.choice(g, min(30, len(g)), replace=False))
        idx = np.array(idx); out = []
        for g in idx:
            m = M[g]; k = int(np.argmax(m == 65535)) if (m == 65535).any() else len(m)
            out.append(([(int(C[g, j]), -1 if m[j] == 361 else int(m[j]) // 19, -1 if m[j] == 361 else int(m[j]) % 19) for j in range(k)], int(hd[g])))
        return out, idx
    if src == "trainsel4":   # extra games (15/player, not in trainsel/trainsel2) for the 5400 players of trainsel+trainsel2
        z = np.load(D + "data/train_arr.npz"); M, C, hd, P = z["M"], z["C"], z["hd"], z["P"]
        used = set(np.load(D + "results/hsl2_trainsel_idx.npy").tolist()) | set(np.load(D + "results/hsl2_trainsel2_idx.npy").tolist())
        pl = np.unique(P[np.array(sorted(used))]); hold = np.random.default_rng(1234).choice(np.unique(P), 800, replace=False); pl = np.setdiff1d(pl, hold)
        order = np.argsort(P, kind="stable"); sp = P[order]; bounds = np.searchsorted(sp, np.arange(P.max() + 2)); r4 = np.random.default_rng(17); idx = []
        for p in pl:
            g = np.array([x for x in order[bounds[p]:bounds[p + 1]] if x not in used]); idx += list(r4.choice(g, min(15, len(g)), replace=False)) if len(g) else []
        idx = np.array(idx); out = []
        for g in idx:
            m = M[g]; k = int(np.argmax(m == 65535)) if (m == 65535).any() else len(m)
            out.append(([(int(C[g, j]), -1 if m[j] == 361 else int(m[j]) // 19, -1 if m[j] == 361 else int(m[j]) % 19) for j in range(k)], int(hd[g])))
        return out, idx
    if src.startswith("pilotlong"):   # pilot: 200 hold-out players x 20 games per shard k (disjoint), for 250-ply extraction
        k = int(src[len("pilotlong"):]); z = np.load(D + "data/train_arr.npz"); M, C, hd, P = z["M"], z["C"], z["hd"], z["P"]
        hold = np.random.default_rng(1234).choice(np.unique(P), 800, replace=False); cnt = np.bincount(P, minlength=P.max() + 1)
        elig = np.array([p for p in hold if cnt[p] >= 20]); pl = np.random.default_rng(99).permutation(elig)[200 * k:200 * (k + 1)]
        order = np.argsort(P, kind="stable"); bounds = np.searchsorted(P[order], np.arange(P.max() + 2)); r = np.random.default_rng(5 + k); idx = []
        for p in pl: g = order[bounds[p]:bounds[p + 1]]; idx += list(r.choice(g, 20, replace=False))
        idx = np.array(idx); out = []
        for g in idx:
            m = M[g]; kk = int(np.argmax(m == 65535)) if (m == 65535).any() else len(m)
            out.append(([(int(C[g, j]), -1 if m[j] == 361 else int(m[j]) // 19, -1 if m[j] == 361 else int(m[j]) % 19) for j in range(kk)], int(hd[g])))
        return out, idx
    if src == 'trainsel2':
        z = np.load(D + 'data/train_arr.npz'); M, C, hd, P = z['M'], z['C'], z['hd'], z['P']
        players = np.unique(P); hold = np.random.default_rng(1234).choice(players, 800, replace=False); rest = np.setdiff1d(players, hold)
        r = np.random.default_rng(7); trp = r.choice(rest, 2400, replace=False); rest2 = np.setdiff1d(rest, trp); r2 = np.random.default_rng(11); trp2 = r2.choice(rest2, min(3000, len(rest2)), replace=False)
        order = np.argsort(P, kind='stable'); sp = P[order]; bounds = np.searchsorted(sp, np.arange(P.max() + 2)); idx = []
        for p in trp2: g = order[bounds[p]:bounds[p + 1]]; idx += list(r2.choice(g, min(25, len(g)), replace=False))
        idx = np.array(idx); out = []
        for g in idx:
            m = M[g]; k = int(np.argmax(m == 65535)) if (m == 65535).any() else len(m)
            out.append(([(int(C[g, j]), -1 if m[j] == 361 else int(m[j]) // 19, -1 if m[j] == 361 else int(m[j]) % 19) for j in range(k)], int(hd[g])))
        return out, idx
    if src == 'hoid':   # ID pilot: 150 eligible hold-out players x 15 games (idx = train_arr row indices), seeds fixed
        z = np.load(D + 'data/train_arr.npz'); M, C, hd, P = z['M'], z['C'], z['hd'], z['P']
        hold = np.random.default_rng(1234).choice(np.unique(P), 800, replace=False); cnt = np.bincount(P, minlength=P.max() + 1)
        elig = np.array([p for p in hold if cnt[p] >= 15]); r = np.random.default_rng(5); pl = r.choice(elig, 150, replace=False)
        order = np.argsort(P, kind='stable'); bounds = np.searchsorted(P[order], np.arange(P.max() + 2)); idx = []
        for p in pl: g = order[bounds[p]:bounds[p + 1]]; idx += list(r.choice(g, 15, replace=False))
        idx = np.array(idx); out = []
        for g in idx:
            m = M[g]; k = int(np.argmax(m == 65535)) if (m == 65535).any() else len(m)
            out.append(([(int(C[g, j]), -1 if m[j] == 361 else int(m[j]) // 19, -1 if m[j] == 361 else int(m[j]) % 19) for j in range(k)], int(hd[g])))
        return out, idx
    if src in ('idtest_pub2', 'idtest_priv2'):   # Task 1 new-family test games; order = candidates (pids sorted, 10 each) then query games; idx = target colour (0 B / 1 W)
        from gocommon import parse
        T = D + 'data/tests1v2/tests/'; tg, cf, qf = ('pub2', 'candidates2_public', 'test2_public') if src.endswith('pub2') else ('priv2', 'candidates2_private', 'test2_private')
        cand = list(csv.DictReader(open(T + f'player_identification_{cf}.csv'))); byp = {}
        for r in cand: byp.setdefault(r['player_id'], []).append(r)
        order = [r for p in sorted(byp) for r in byp[p]]; out = []; cols = []
        for r in order: h, mv = parse(r['sgf_content']); out.append((mv, h)); cols.append(0 if r['color'] == 'B' else 1)
        for r in csv.DictReader(open(T + f'player_identification_{qf}.csv')):
            for i in range(1, int(r['num_games']) + 1): h, mv = parse(r[f'sgf_{i}']); out.append((mv, h)); cols.append(0 if r[f'color_{i}'] == 'B' else 1)
        return out, np.array(cols)
    if src == 'test2':
        from gocommon import parse
        out = []
        for nm in ('public', 'private'):
            for r in csv.DictReader(open(D + f'data/tests2/tests/rank_prediction_test_{nm}.csv')):
                for i in range(1, int(r['num_games']) + 1):
                    h, mv = parse(r[f'sgf_{i}']); out.append((mv, h))
        return out, np.arange(len(out))
    raise ValueError(src)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--n', type=int, default=200); ap.add_argument('--maxm', type=int, default=160); ap.add_argument('--nproc', type=int, default=16); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=1)
    a = ap.parse_args()
    dev = torch.device('cuda', 0); t = torch.arange(32, device=dev, dtype=torch.float32); assert (t * t).sum().item() == 10416; print('REAL_TENSOR_PASS', flush=True)
    from katago.train.load_model import load_model
    from katago.game.sgfmetadata import SGFMetadata
    import datetime
    model, swa, _ = load_model(D + 'katago/humanv0.ckpt', use_swa=False, device=dev, pos_len=19, verbose=False)
    model.eval(); cfg = model.config
    rows = []
    for p in PROFILES:
        m = SGFMetadata(inverseBRank=INV[p], inverseWRank=INV[p], bIsHuman=True, wIsHuman=True, gameRatednessIsUnknown=True, tcIsUnknown=False, tcIsByoYomi=True, mainTimeSeconds=1200, periodTimeSeconds=30, byoYomiPeriods=5, gameDate=datetime.date(2020, 3, 1), source=SGFMetadata.SOURCE_KGS)
        rows.append(m.get_metadata_row('b', 361))
    meta = torch.tensor(np.stack(rows), dtype=torch.float32, device=dev); R = len(PROFILES)
    games, idx = games_from_src(a.src, a.n); games, idx = games[a.shard::a.nshard], idx[a.shard::a.nshard]; G = len(games); W = 3 * R + 3; out = np.full((G, a.maxm - a.start, W), np.nan, np.float16); out[:, :, W - 1] = -1
    _init()
    t0 = time.time()
    with mp.Pool(a.nproc, initializer=_init) as pool:
        for gi, res in enumerate(pool.imap(worker, [(mv, hd, a.maxm, cfg, a.start) for mv, hd in games], chunksize=4)):
            if res is None: continue
            b, g, tg, mo = res; n = len(tg)
            ent = np.zeros((n, R), np.float32); t1 = np.zeros((n, R), np.float32); lp = np.zeros((n, R), np.float32); win = np.zeros(n, np.float32); lead = np.zeros(n, np.float32)
            with torch.no_grad():
                for s in range(0, n, 64):
                    bb = torch.from_numpy(b[s:s + 64].astype(np.float32)).to(dev); gg = torch.from_numpy(g[s:s + 64].astype(np.float32)).to(dev); k = len(bb)
                    bb = bb.repeat_interleave(R, 0); gg = gg.repeat_interleave(R, 0); mm = meta.repeat(k, 1)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        o = model(bb, gg, input_meta=mm)
                    outs = model.postprocess_output(o)[0]
                    pl = outs[0].float(); vl = outs[1].float(); ld = outs[10].float()
                    lpp = torch.log_softmax(pl[:, 0, :], -1).view(k, R, -1)
                    tt = torch.from_numpy(tg[s:s + 64].astype(np.int64)).to(dev)
                    pr = lpp.exp(); ent[s:s + k] = (-(pr * lpp).sum(-1)).cpu().numpy(); t1[s:s + k] = (lpp.argmax(-1) == tt[:, None]).float().cpu().numpy()
                    lp[s:s + k] = lpp.gather(2, tt[:, None, None].expand(k, R, 1))[:, :, 0].cpu().numpy()
                    win[s:s + k] = torch.softmax(vl, -1)[:, 0].view(k, R)[:, 4].cpu().numpy(); lead[s:s + k] = ld.view(k, R)[:, 4].cpu().numpy()
            out[gi, :n, :R] = lp; out[gi, :n, R:2 * R] = ent; out[gi, :n, 2 * R:3 * R] = t1; out[gi, :n, 3 * R] = win; out[gi, :n, 3 * R + 1] = lead; out[gi, :n, 3 * R + 2] = mo
            if gi % 50 == 0: print(gi, G, f'{time.time() - t0:.0f}s', flush=True)
    np.save(a.out, out); np.save(a.out.replace('.npy', '_idx.npy'), idx); print('DONE', time.time() - t0, flush=True)
