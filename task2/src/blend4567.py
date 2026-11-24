import sys,math,numpy as np,torch,torch.nn as nn
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/results/')
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
from hsl_rank3c import metric
z=np.load(D+'hslr3_val.npz');pv,Lv,Yv,bags,idx=z['pv'],z['Lv'],z['Yv'],z['bags'],z['idx']
ftl=np.load(D+'ftlp_trainsel_4567.npy');tsi=np.load(D+'trainsel_idx.npy');fm={g:i for i,g in enumerate(tsi)}
Ln=np.array([ftl[[fm[idx[j]] for j in b[b>=0]]].sum(0) for b in bags])
print('ft45 alone',metric(Lv,Yv),'ft4567 alone',metric(Ln,Yv))
bt=sys.argv[1].split(",");st=sys.argv[2].split(",");lgs=[]
for tag in bt:
    Fh=np.load(D+f'ftbag_{tag}_holdfeat.npy').astype(np.float32);d=Fh.shape[1]//2;head=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,10));head.load_state_dict(torch.load(D+f'ftbag_{tag}_head.pt',map_location='cpu'));head.eval()
    E=np.stack([Fh[b[b>=0]].mean(0) for b in bags])
    with torch.no_grad():lgs.append(torch.log_softmax(head(torch.from_numpy(E)),-1).numpy())
lg=np.mean(lgs,0);ls=np.mean([np.load(D+f'hslseq_{t}_valLP.npy') for t in st],0)
for nm,L in (('old',Lv),('new',Ln)):
    r=sorted(((round(metric(a*L+b*pv+c*lg+dd*ls,Yv)[0],4),a,b,c,dd) for a in (0,.25,.5,.75,1) for b in (0,.5,1) for c in (0,.5,1) for dd in (.5,1,2,3)))[::-1][:4]
    print(nm,r)
