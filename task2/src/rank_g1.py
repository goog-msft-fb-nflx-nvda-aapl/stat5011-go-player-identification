import sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
from train_lm import LM,sym_tables,D
tag=sys.argv[1]
dev=torch.device('cuda',0);t=torch.arange(32,device=dev,dtype=torch.float32);assert (t*t).sum().item()==10416
ck=torch.load(D+f'results/ftbag_{tag}.pt',map_location='cpu',weights_only=False);a=ck['args'];lck=torch.load(D+f"results/lm_{a['init']}.pt",map_location='cpu',weights_only=False);la=lck['args'];d=la['d']
lm=LM(la['d'],la['layers'],la['heads']);head=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,10));sd=ck['net']
lm.load_state_dict({k[3:]:v for k,v in sd.items() if k.startswith('lm.')});head.load_state_dict({k[5:]:v for k,v in sd.items() if k.startswith('head.')});lm=lm.to(dev).eval();head=head.to(dev).eval();SYM=sym_tables(dev)
z=np.load(D+'data/train_arr.npz');M=z['M'];C=z['C'];hd=z['hd'];col=z['col'];P=z['P']
idx=np.concatenate([np.load(D+'results/hsl2_trainsel_idx.npy'),np.load(D+'results/hsl2_trainsel2_idx.npy')]);hold=np.random.default_rng(1234).choice(np.unique(P),800,replace=False);pos=np.flatnonzero(np.isin(P[idx],hold));G=idx[pos]
out=np.zeros((len(idx),2*d),np.float16)
@torch.no_grad()
def feats(ib,tta=4):
    p0=torch.from_numpy(M[ib][:,:320].astype(np.int64)).to(dev);p0[p0==65535]=362;c=torch.from_numpy(C[ib][:,:320].astype(np.int64)).to(dev);h_=torch.from_numpy(np.minimum(hd[ib],9).astype(np.int64)).to(dev);tt=torch.from_numpy(col[ib].astype(np.int64)).to(dev);B=len(ib);L=320;acc=0
    for k in range(tta):
        p=torch.gather(SYM[k].expand(B,-1),1,p0);ply=torch.arange(L,device=dev)[None].expand(B,L);cond=torch.full((B,),10,device=dev)
        x=lm.pos(p)+lm.col(c)+lm.ply(ply)+(lm.hd(h_)+lm.tgt(tt)+lm.cond(cond))[:,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):h=lm.ln(lm.enc(x,mask=torch.nn.Transformer.generate_square_subsequent_mask(L,device=dev),is_causal=True))
        h=h.float();own=((c==tt[:,None])&(p!=362)).float().unsqueeze(-1);al=(p!=362).float().unsqueeze(-1)
        acc=acc+torch.cat([(h*own).sum(1)/own.sum(1).clamp(min=1),(h*al).sum(1)/al.sum(1).clamp(min=1)],1)
    return (acc/tta).cpu().numpy()
for s in (range(0,len(G),256) if not __import__('os').environ.get('SKIPHOLD') else []):out[pos[s:s+256]]=feats(G[s:s+256]).astype(np.float16)
(np.save(D+f'results/ftbag_{tag}_holdfeat.npy',out) if not __import__('os').environ.get('SKIPHOLD') else None);torch.save(head.state_dict(),D+f'results/ftbag_{tag}_head.pt');print('done',len(G))
# test rows
import csv;from gocommon import parse
csv.field_size_limit(10**9);T=D+'data/tests2/tests/';LPs=[]
for nm in ('public','private'):
    for r in csv.DictReader(open(T+f'rank_prediction_test_{nm}.csv')):
        n=int(r['num_games']);pos=np.full((n,320),362,np.int64);cc=np.zeros((n,320),np.int64);hh=np.zeros(n,np.int64);tg=np.array([0 if r[f'color_{i}']=='B' else 1 for i in range(1,n+1)])
        for i in range(n):
            h,mv=parse(r[f'sgf_{i+1}']);hh[i]=min(h,9)
            for j,(c,x,y) in enumerate(mv[:320]):pos[i,j]=361 if x<0 else x*19+y;cc[i,j]=c
        # reuse feats via manual arrays
        acc=0
        with torch.no_grad():
            p0=torch.from_numpy(pos).to(dev);c=torch.from_numpy(cc).to(dev);h_=torch.from_numpy(hh).to(dev);tt=torch.from_numpy(tg).to(dev);B=n;L=320
            for k in range(4):
                p=torch.gather(SYM[k].expand(B,-1),1,p0);ply=torch.arange(L,device=dev)[None].expand(B,L);cond=torch.full((B,),10,device=dev)
                x=lm.pos(p)+lm.col(c)+lm.ply(ply)+(lm.hd(h_)+lm.tgt(tt)+lm.cond(cond))[:,None]
                with torch.autocast('cuda',dtype=torch.bfloat16):h=lm.ln(lm.enc(x,mask=torch.nn.Transformer.generate_square_subsequent_mask(L,device=dev),is_causal=True))
                h=h.float();own=((c==tt[:,None])&(p!=362)).float().unsqueeze(-1);al=(p!=362).float().unsqueeze(-1)
                acc=acc+torch.cat([(h*own).sum(1)/own.sum(1).clamp(min=1),(h*al).sum(1)/al.sum(1).clamp(min=1)],1)
            e=(acc/4).mean(0,keepdim=True);LPs.append(torch.log_softmax(head(e),-1)[0].cpu().numpy())
np.save(D+f'results/ftbag_{tag}_test_lp.npy',np.array(LPs));print('test done',len(LPs))
