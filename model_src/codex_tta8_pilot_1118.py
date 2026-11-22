"""Embed ALL games of the 800 hold-out players (same hold-out as train_idft*.py: rng 1234) with hold-out idft models -> results/HO_<tag>.npy (float16, N x 256, 4-sym TTA mean) and results/HO_meta.npz (game index into train_arr, player id, target colour).
usage: CUDA_VISIBLE_DEVICES=1 python holdout_embeds.py a2 b1 b2 b3 h5"""
import sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
from train_lm import LM,sym_tables,D
dev=torch.device('cuda',0);t=torch.arange(32,device=dev,dtype=torch.float32);assert (t*t).sum().item()==10416
z=np.load(D+'data/train_arr.npz');M=z['M'];C=z['C'];hd=z['hd'];col=z['col'];P=z['P']
hold=np.random.default_rng(1234).choice(np.unique(P),800,replace=False);cnt=np.bincount(P,minlength=P.max()+1)
elig=np.array([p for p in hold if cnt[p]>=15]);idx=np.flatnonzero(np.isin(P,elig));meta=np.load(D+'results/HO_meta.npz');assert all(np.array_equal(meta[key],value) for key,value in {'idx':idx,'P':P[idx],'col':col[idx],'hd':hd[idx]}.items());print('games',len(idx),'players',len(elig),flush=True)
rng=np.random.default_rng(1818)
selected_players=rng.choice(np.unique(P[idx]),150,replace=False)
selected_rows=np.concatenate([rng.choice(np.flatnonzero(P[idx]==p),15,replace=False) for p in selected_players])
np.save(D+'results/codex_tta8_pilot_rows_1118.npy',selected_rows)
idx=idx[selected_rows]
print('TTA pilot games',len(idx),flush=True)
SYM=sym_tables(dev)
for tag in sys.argv[1:]:
    ck=torch.load(D+f'results/idft_{tag}.pt',map_location='cpu',weights_only=False);a=ck['args'];assert not a.get('allp',0)
    lck=torch.load(D+f"results/lm_{a['init']}.pt",map_location='cpu',weights_only=False);la=lck['args'];d=la['d']
    lm=LM(la['d'],la['layers'],la['heads']);proj=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,1024),nn.GELU(),nn.Linear(1024,256))
    sd=ck['net'];lm.load_state_dict({k[3:]:v for k,v in sd.items() if k.startswith('lm.')});proj.load_state_dict({k[5:]:v for k,v in sd.items() if k.startswith('proj.')});lm=lm.to(dev).eval();proj=proj.to(dev).eval()
    out=np.zeros((len(idx),256),np.float16);L=320
    with torch.no_grad():
        for s in range(0,len(idx),128):
            ib=idx[s:s+128];p0=torch.from_numpy(M[ib,:L].astype(np.int64)).to(dev);p0[p0==65535]=362;c=torch.from_numpy(C[ib,:L].astype(np.int64)).to(dev);h_=torch.from_numpy(np.minimum(hd[ib],9).astype(np.int64)).to(dev);tt=torch.from_numpy(col[ib].astype(np.int64)).to(dev);B=len(ib);acc=0
            for k in range(8):
                p=torch.gather(SYM[k].expand(B,-1),1,p0);ply=torch.arange(L,device=dev)[None].expand(B,L);cond=torch.full((B,),10,device=dev)
                x=lm.pos(p)+lm.col(c)+lm.ply(ply)+(lm.hd(h_)+lm.tgt(tt)+lm.cond(cond))[:,None]
                with torch.autocast('cuda',dtype=torch.bfloat16):h=lm.ln(lm.enc(x,mask=torch.nn.Transformer.generate_square_subsequent_mask(L,device=dev),is_causal=True))
                h=h.float();own=((c==tt[:,None])&(p!=362)).float().unsqueeze(-1);al=(p!=362).float().unsqueeze(-1)
                pooled=torch.cat([(h*own).sum(1)/own.sum(1).clamp(min=1),(h*al).sum(1)/al.sum(1).clamp(min=1)],1);acc=acc+F.normalize(proj(pooled),dim=-1)
            out[s:s+B]=F.normalize(acc,dim=-1).cpu().numpy().astype(np.float16)
    assert np.isfinite(out).all();np.save(D+f'results/HO_{tag}_tta8pilot1118.npy',out);print('done',tag,flush=True)
