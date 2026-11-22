"""Rank-conditioned causal move LM. p(move_{t+1} | moves<=t, rank, hd, target colour); loss on target-colour moves only.
Rank inference = Bayes over 10 rank conditions (sum log p over the player's own moves).
usage: CUDA_VISIBLE_DEVICES=3 python train_lm.py <tag> [--epochs --d --layers --maxlen --hold --noaug]"""
import os, sys, time, argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
D = '/home/jtan/solardefect/go_20261113/'
ap = argparse.ArgumentParser(); ap.add_argument('tag'); ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--d', type=int, default=384)
ap.add_argument('--layers', type=int, default=8); ap.add_argument('--heads', type=int, default=6); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=4e-4)
ap.add_argument('--maxlen', type=int, default=320); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--hold', type=int, default=800); ap.add_argument('--noaug', action='store_true')
ap.add_argument('--cond_drop', type=float, default=0.0)
if __name__ == '__main__':
    a = ap.parse_args()
else:
    a = ap.parse_args([ 'x' ])

class LM(nn.Module):
    def __init__(s, d, layers, heads):
        super().__init__()
        s.pos = nn.Embedding(363, d); s.col = nn.Embedding(2, d); s.ply = nn.Embedding(512, d)
        s.cond = nn.Embedding(10 + 1, d)  # rank (+ null)
        s.hd = nn.Embedding(10, d); s.tgt = nn.Embedding(2, d)
        L = nn.TransformerEncoderLayer(d, heads, d * 4, 0.0, batch_first=True, norm_first=True, activation='gelu')
        s.enc = nn.TransformerEncoder(L, layers); s.ln = nn.LayerNorm(d); s.out = nn.Linear(d, 362)
    def forward(s, pos, col, hd, tgt, cond):
        # input at t: move t (pos, col) ; predict move t+1.  pos (B,L) with pad=362 mapped to emb idx
        B, L = pos.shape
        ply = torch.arange(L, device=pos.device)[None].expand(B, L)
        x = s.pos(pos) + s.col(col) + s.ply(ply) + (s.hd(hd) + s.tgt(tgt) + s.cond(cond))[:, None]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L, device=pos.device)
        h = s.ln(s.enc(x, mask=mask, is_causal=True))
        return s.out(h)

def sym_tables(dev):
    xs, ys = np.meshgrid(np.arange(19), np.arange(19), indexing='ij')
    def tab(f):
        X, Y = f(xs, ys); T = np.full(363, 362); T[:361] = (X * 19 + Y).ravel(); T[361] = 361; return torch.from_numpy(T).to(dev)
    fs = [lambda x, y: (x, y), lambda x, y: (18 - x, y), lambda x, y: (x, 18 - y), lambda x, y: (18 - x, 18 - y), lambda x, y: (y, x), lambda x, y: (18 - y, x), lambda x, y: (y, 18 - x), lambda x, y: (18 - y, 18 - x)]
    return torch.stack([tab(f) for f in fs])

def lm_loglik(net, pos, col, hd, tgt, cond):
    """per-game sum log p over target-colour moves; pos (B,L) pad=362"""
    inp = pos[:, :-1]; tgt_pos = pos[:, 1:]; ccol = col[:, :-1]; nxt_col = col[:, 1:]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lg = net(inp, ccol, hd, tgt, cond)
    lp = F.log_softmax(lg.float(), -1)
    valid = (tgt_pos != 362) & (nxt_col == tgt[:, None])
    ll = lp.gather(2, tgt_pos.clamp(max=361)[..., None]).squeeze(-1)
    return (ll * valid).sum(1), valid.sum(1)

if __name__ == '__main__':
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda', 0); t = torch.arange(32, device=dev, dtype=torch.float32); assert (t * t).sum().item() == 10416; print('REAL_TENSOR_PASS', flush=True)
    z = np.load(D + 'data/train_arr.npz'); M = z['M']; C = z['C']; hd = z['hd']; y = z['y']; col = z['col']; P = z['P']
    rng = np.random.default_rng(1234); players = np.unique(P); hold = rng.choice(players, a.hold, replace=False); ishold = np.isin(P, hold)
    tr = np.flatnonzero(~ishold)
    Mg = torch.from_numpy(M[:, :a.maxlen].astype(np.int64)).to(dev); Cg = torch.from_numpy(C[:, :a.maxlen].astype(np.int64)).to(dev); Mg[Mg == 65535] = 362
    hdg = torch.from_numpy(np.minimum(hd, 9).astype(np.int64)).to(dev); yg = torch.from_numpy(y.astype(np.int64)).to(dev); colg = torch.from_numpy(col.astype(np.int64)).to(dev)
    SYM = sym_tables(dev)
    net = LM(a.d, a.layers, a.heads).to(dev); print('params', sum(p.numel() for p in net.parameters()) / 1e6, 'M', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.98))
    steps_per = len(tr) // a.bs; total = steps_per * a.epochs; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.03)
    trg = torch.from_numpy(tr).to(dev)
    def val():
        va = np.flatnonzero(ishold); sub = torch.from_numpy(va[np.random.default_rng(0).permutation(len(va))[:3000]]).to(dev); net.eval(); acc = 0; ex = 0; nll_true = 0; cnt = 0; probs = []
        with torch.no_grad():
            for s in range(0, len(sub), 250):
                ib = sub[s:s + 250]; pos = Mg[ib]; lls = []
                for r in range(10):
                    ll, n = lm_loglik(net, pos, Cg[ib], hdg[ib], colg[ib], torch.full((len(ib),), r, device=dev)); lls.append(ll)
                L_ = torch.stack(lls, 1); pred = L_.argmax(1); ex += (pred == yg[ib]).sum().item(); cnt += len(ib)
        net.train(); return ex / cnt
    t0 = time.time(); net.train()
    for ep in range(a.epochs):
        perm = trg[torch.randperm(len(trg), device=dev)]; tl = 0; tn = 0
        for s in range(steps_per):
            ib = perm[s * a.bs:(s + 1) * a.bs]; pos = Mg[ib]
            if not a.noaug:
                k = torch.randint(0, 8, (len(ib),), device=dev); pos = torch.gather(SYM[k], 1, pos)
            cond = yg[ib]
            ll, n = lm_loglik(net, pos, Cg[ib], hdg[ib], colg[ib], cond)
            loss = -(ll.sum() / n.sum().clamp(min=1))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item(); tn += 1
            if s % 500 == 0: print(f'ep{ep} step{s}/{steps_per} nll/move {tl / tn:.4f} t={time.time() - t0:.0f}s', flush=True)
        ex = val(); print(f'EPOCH {ep} nll {tl / tn:.4f} val_game_exact(bayes) {ex:.4f} t={time.time() - t0:.0f}s', flush=True)
        torch.save({'net': net.state_dict(), 'args': vars(a), 'hold': hold}, D + f'results/lm_{a.tag}.pt')
    print('DONE', flush=True)
