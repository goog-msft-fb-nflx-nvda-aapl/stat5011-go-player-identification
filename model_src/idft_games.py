"""[per-game variant] Embed test pools with an idft model -> results/Sidft_{tag}_{pool}.npy (nq x ncand cosine of query-mean vs candidate-mean, TTA over 4 syms)."""
import sys,csv,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
tag=sys.argv[1];import types
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/');T=D+'data/tests1v2/tests/';csv.field_size_limit(10**9)
from gocommon import parse
from train_lm import LM,sym_tables
dev=torch.device('cuda',0);t=torch.arange(32,device=dev,dtype=torch.float32);assert (t*t).sum().item()==10416
ck=torch.load(D+f'results/idft_{tag}.pt',map_location='cpu',weights_only=False);a=ck['args']
lck=torch.load(D+f"results/lm_{a['init']}.pt",map_location='cpu',weights_only=False);la=lck['args']
import torch.nn as nn
lm=LM(la['d'],la['layers'],la['heads']);d=la['d'];proj=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,1024),nn.GELU(),nn.Linear(1024,256))
sd=ck['net'];lm.load_state_dict({k[3:]:v for k,v in sd.items() if k.startswith('lm.')});proj.load_state_dict({k[5:]:v for k,v in sd.items() if k.startswith('proj.')});lm=lm.to(dev).eval();proj=proj.to(dev).eval();SYM=sym_tables(dev)
@torch.no_grad()
def embed(sgfs,cols,tta=4):
    N=len(sgfs);pos=np.full((N,320),362,np.int64);cc=np.zeros((N,320),np.int64);hd=np.zeros(N,np.int64);tg=np.array([0 if c=='B' else 1 for c in cols],np.int64)
    for i,s in enumerate(sgfs):
        h,mv=parse(s);hd[i]=min(h,9)
        for j,(c,x,y) in enumerate(mv[:320]):pos[i,j]=361 if x<0 else x*19+y;cc[i,j]=c
    out=[];L=320
    for s in range(0,N,256):
        p0=torch.from_numpy(pos[s:s+256]).to(dev);c=torch.from_numpy(cc[s:s+256]).to(dev);h_=torch.from_numpy(hd[s:s+256]).to(dev);tt=torch.from_numpy(tg[s:s+256]).to(dev);B=len(p0);acc=0
        for k in range(tta):
            p=torch.gather(SYM[k].expand(B,-1),1,p0);ply=torch.arange(L,device=dev)[None].expand(B,L);cond=torch.full((B,),10,device=dev)
            x=lm.pos(p)+lm.col(c)+lm.ply(ply)+(lm.hd(h_)+lm.tgt(tt)+lm.cond(cond))[:,None]
            with torch.autocast('cuda',dtype=torch.bfloat16):h=lm.ln(lm.enc(x,mask=torch.nn.Transformer.generate_square_subsequent_mask(L,device=dev),is_causal=True))
            h=h.float();own=((c==tt[:,None])&(p!=362)).float().unsqueeze(-1);al=(p!=362).float().unsqueeze(-1)
            pooled=torch.cat([(h*own).sum(1)/own.sum(1).clamp(min=1),(h*al).sum(1)/al.sum(1).clamp(min=1)],1);acc=acc+F.normalize(proj(pooled),dim=-1)
        out.append(F.normalize(acc,dim=-1).cpu())
    return torch.cat(out).numpy()
def nrm(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
for tg,cf,qf in (('pub2','candidates2_public','test2_public'),('priv2','candidates2_private','test2_private')):
    cand=list(csv.DictReader(open(T+f'player_identification_{cf}.csv')));byp={}
    for r in cand:byp.setdefault(r['player_id'],[]).append(r)
    pids=sorted(byp);order=[r for p in pids for r in byp[p]];assert all(len(byp[p])==10 for p in pids)
    Ec=embed([r['sgf_content'] for r in order],[r['color'] for r in order]).reshape(len(pids),10,-1)
    qs=list(csv.DictReader(open(T+f'player_identification_{qf}.csv')));sg=[];cl=[];qi=[]
    for j,r in enumerate(qs):
        n=int(r['num_games']);sg+= [r[f'sgf_{i}'] for i in range(1,n+1)];cl+=[r[f'color_{i}'] for i in range(1,n+1)];qi+=[j]*n
    Eq=embed(sg,cl);np.save(D+f'results/G_{tag}_{tg}_cand.npy',Ec.astype(np.float16));np.save(D+f'results/G_{tag}_{tg}_q.npy',Eq.astype(np.float16));np.save(D+f'results/G_{tag}_{tg}_qidx.npy',np.array(qi));print(tg,Ec.shape,Eq.shape,flush=True)
print('DONE',flush=True)
