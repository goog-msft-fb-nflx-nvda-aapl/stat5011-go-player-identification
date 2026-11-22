"""Question-level rank sim for LM checkpoint: per-game 10-rank loglik (TTA over syms), aggregate over n games, expected-score decision. Saves per-game loglik for held-out games."""
import sys,math,numpy as np,torch
sys.path.insert(0,'/home/jtan/solardefect/go_20261113/src')
from train_lm import LM,lm_loglik,sym_tables,D
tag=sys.argv[1];tta=int(sys.argv[2]) if len(sys.argv)>2 else 1;dev=torch.device('cuda',0)
ck=torch.load(D+f'results/lm_{tag}.pt',map_location='cpu',weights_only=False);a=ck['args'];net=LM(a['d'],a['layers'],a['heads']);net.load_state_dict(ck['net']);net=net.to(dev).eval();sym=sym_tables(dev)
z=np.load(D+'data/train_arr.npz');M,C,hd,y,col,P=z['M'],z['C'],z['hd'],z['y'],z['col'],z['P'];hold=ck['hold'];va=np.flatnonzero(np.isin(P,hold))
byp={}
for i in va:byp.setdefault((P[i],col[i]),[]).append(i)
keys=[k for k,v in byp.items() if len(v)>=20];rng=np.random.default_rng(3)
need=sorted({i for k in keys for i in byp[k]});need=np.array(need)
pos=torch.from_numpy(M[need,:a['maxlen']].astype(np.int64)).to(dev);pos[pos==65535]=362;cc=torch.from_numpy(C[need,:a['maxlen']].astype(np.int64)).to(dev);h=torch.from_numpy(np.minimum(hd[need],9).astype(np.int64)).to(dev);tg=torch.from_numpy(col[need].astype(np.int64)).to(dev)
LL=np.zeros((len(need),10),np.float32)
with torch.no_grad():
    for s in range(0,len(need),250):
        sl=slice(s,s+250);acc=0
        for k in range(tta):
            pp=torch.gather(sym[k].expand(len(pos[sl]),-1),1,pos[sl]);cols=[]
            for r in range(10):
                ll,_=lm_loglik(net,pp,cc[sl],h[sl],tg[sl],torch.full((len(pp),),r,device=dev));cols.append(ll)
            acc=acc+torch.stack(cols,1)
        LL[s:s+250]=(acc/tta).cpu().numpy()
np.save(D+f'results/lm_{tag}_holdLL.npy',LL);np.save(D+f'results/lm_{tag}_holdidx.npy',need)
loc={g:i for i,g in enumerate(need)};yv=y
def qs(lp,t):
    p=np.exp(lp-lp.max());p/=p.sum();sc=p.copy();sc[1:]+=math.exp(-1)*p[:-1];sc[:-1]+=math.exp(-1)*p[1:];k=sc.argmax();return 1. if k==t else (math.exp(-1) if abs(k-t)==1 else 0.)
print('game exact',np.mean(LL.argmax(1)==yv[need]));tot=[]
for n in (5,10,15,20):
    s=[]
    for _ in range(800):
        k=keys[rng.integers(len(keys))];ix=rng.choice(byp[k],n,replace=False);s.append(qs(LL[[loc[i] for i in ix]].sum(0),yv[ix[0]]))
    print('RANK n',n,'%.4f'%np.mean(s));tot+=s
print('RANK ALL %.4f'%np.mean(tot))
