import sys,math,numpy as np,torch,torch.nn as nn
D='/home/jtan/solardefect/go_20261113/results/'
sys.path.insert(0,'/home/jtan/solardefect/go_20261113/src')
from hsl_rank3c import metric
z=np.load(D+'hslr3_val.npz');pv,Lv,Yv,bags,idx=z['pv'],z['Lv'],z['Yv'],z['bags'],z['idx']
ftl=np.load(D+'ftlp_trainsel_4567.npy');tsi=np.load(D+'trainsel_idx.npy');fm={g:i for i,g in enumerate(tsi)}
Ln=np.array([ftl[[fm[idx[j]] for j in b[b>=0]]].sum(0) for b in bags]);lgs=[]
for tag in ['g1','g2','g3','g4','g6','g7']:
    Fh=np.load(D+f'ftbag_{tag}_holdfeat.npy').astype(np.float32);d=Fh.shape[1]//2;head=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,10));head.load_state_dict(torch.load(D+f'ftbag_{tag}_head.pt',map_location='cpu'));head.eval()
    E=np.stack([Fh[b[b>=0]].mean(0) for b in bags])
    with torch.no_grad():lgs.append(torch.log_softmax(head(torch.from_numpy(E)),-1).numpy())
lg=np.mean(lgs,0);ls=np.mean([np.load(D+f'hslseq_{t}_valLP.npy') for t in ('s1','s2','s3')],0)
L=.25*Ln+.5*pv+lg+2*ls
print('class counts',np.bincount(Yv),'base',metric(L,Yv))
def norm(L):L=L-L.max(1,keepdims=True);p=np.exp(L);return p/p.sum(1,keepdims=True)
for T in (1,2,3,4,6):
  for it in (1,3,10,30):
    p=norm(L/T);w=np.ones(10)
    for _ in range(it):
        q=p*w;q/=q.sum(1,keepdims=True);w=w/(q.mean(0)*10)
    q=p*w;print('T',T,'it',it,'%.4f'%metric(np.log(q/q.sum(1,keepdims=True)+1e-12),Yv)[0])
