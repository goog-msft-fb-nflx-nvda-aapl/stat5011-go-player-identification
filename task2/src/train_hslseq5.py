"""Bag-level rank model on HumanSL per-move feature sequences (hsl2 outputs). usage: train_hslseq.py <tag> [--epochs N]"""
import sys,time,math,csv,argparse,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/');csv.field_size_limit(10**9)
ap=argparse.ArgumentParser();ap.add_argument('tag');ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--d',type=int,default=128);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--steps',type=int,default=400);ap.add_argument('--lr',type=float,default=5e-4)
a=ap.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed)
dev=torch.device('cuda',0);t=torch.arange(32,device=dev,dtype=torch.float32);assert (t*t).sum().item()==10416
z=np.load(D+'data/train_arr.npz');P,y,col,hdv=z['P'],z['y'],z['col'],z['hd']
import hslcat
SR=__import__('os').environ.get('SRCS','trainsel,trainsel2').split(',');idx=np.concatenate([np.load(D+f'results/hsl2_{q}_idx.npy') for q in SR])
R=8
def prep(o,c,hd):
    """o (G,100,27) float16 -> (G,100,30) float32 with NaN->0, perspective-corrected, + masks"""
    o=np.asarray(o,np.float32);val=~np.isnan(o[:,:,0]);mv=o[:,:,3*R+2];own=(val&(mv==c[:,None])).astype(np.float32);x=np.nan_to_num(o[:,:,:3*R+2]).copy()
    flip=(val&(mv!=c[:,None]))
    x[:,:,3*R]=np.where(flip,1-x[:,:,3*R],x[:,:,3*R]);x[:,:,3*R+1]=np.where(flip,-x[:,:,3*R+1],x[:,:,3*R+1])
    return np.concatenate([x,own[:,:,None],val[:,:,None].astype(np.float32),np.broadcast_to((hd[:,None,None]/9.0).astype(np.float32),(len(x),x.shape[1],1))],2)
Xs=[];mu=sd=None
for q in SR:
    ixq=np.load(D+f'results/hsl2_{q}_idx.npy');Xq=prep(hslcat.load_full(q),col[ixq],hdv[ixq]);mq=(Xq[:,:,3*R+3]>0)
    if mu is None:mu=Xq[:40000][mq[:40000]][:,:3*R+2].mean(0);sd=Xq[:40000][mq[:40000]][:,:3*R+2].std(0)+1e-6
    Xq[:,:,:3*R+2]=np.where(mq[:,:,None],(Xq[:,:,:3*R+2]-mu)/sd,0);Xs.append(Xq.astype(np.float16));del Xq;print(q,len(ixq),flush=True)
Xall=np.concatenate(Xs);del Xs
pass
Xg=torch.from_numpy(Xall).to(dev);Pi=P[idx];yi=y[idx]
hold=np.random.default_rng(1234).choice(np.unique(P),800,replace=False);ish=np.isin(Pi,hold)
def byplayer(sel):
    b={}
    for g in sel:b.setdefault(Pi[g],[]).append(g)
    return {p:np.array(v) for p,v in b.items() if len(v)>=5}
Btr=byplayer(np.flatnonzero(~ish));ptr=list(Btr)
Cin=3*R+5;d=a.d
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.inp=nn.Linear(Cin,d);s.pos=nn.Parameter(torch.zeros(hslcat.NP,d));l=nn.TransformerEncoderLayer(d,4,2*d,0.1,batch_first=True,norm_first=True);s.enc=nn.TransformerEncoder(l,3);s.ln=nn.LayerNorm(d);s.head=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.GELU(),nn.Dropout(0.1),nn.Linear(d,10))
    def game(s,x):
        v=x[:,:,3*R+3]>0;h=s.ln(s.enc(s.inp(x)+s.pos,src_key_padding_mask=~v));own=(x[:,:,3*R+2]>0.5).float().unsqueeze(-1);al=v.float().unsqueeze(-1)
        return torch.cat([(h*own).sum(1)/own.sum(1).clamp(min=1),(h*al).sum(1)/al.sum(1).clamp(min=1)],1)
net=Net().to(dev);opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=0.05);total=a.epochs*a.steps;sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=total,pct_start=0.1)
from hsl_rank3 import metric
vz=np.load(D+'results/hslr3_val.npz');bags=vz['bags'];Yv=vz['Yv'];pv=vz['pv'];Lv=vz['Lv']
@torch.no_grad()
def embed_all(pos_list):
    net.eval();out=np.zeros((len(Xg),2*d),np.float32)
    for s in range(0,len(pos_list),512):
        ib=torch.from_numpy(pos_list[s:s+512]).to(dev);out[pos_list[s:s+512]]=net.game(Xg[ib].float()).cpu().numpy()
    net.train();return out
def val():
    ps=np.unique(bags[bags>=0]);Em=embed_all(ps);E=np.stack([Em[b[b>=0]].mean(0) for b in bags])
    with torch.no_grad():lp=torch.log_softmax(net.head(torch.from_numpy(E).to(dev)),-1).cpu().numpy()
    return lp
t0=time.time();net.train()
for ep in range(a.epochs):
    tl=0
    for s in range(a.steps):
        NP=32;n=np.random.randint(5,13);ps=np.random.choice(len(ptr),NP,replace=False);ib=np.concatenate([np.random.choice(Btr[ptr[i]],n,replace=len(Btr[ptr[i]])<n) for i in ps]);yt=torch.tensor([int(yi[Btr[ptr[i]][0]]) for i in ps],device=dev)
        g=net.game(Xg[torch.from_numpy(ib).to(dev)].float());o=net.head(g.view(NP,n,-1).mean(1)).float();r=torch.arange(10,device=dev)[None].float();tg=F.softmax(-(r-yt[:,None].float()).abs()/1.0,-1);loss=-(tg*F.log_softmax(o,-1)).sum(1).mean()
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();sched.step();tl+=loss.item()
    if ep%3==2 or ep==a.epochs-1:
        lp=val();print(f'EPOCH {ep} loss {tl/a.steps:.4f} val {metric(lp,Yv)} blend {metric(0.5*Lv+pv+lp,Yv)[0]:.4f} t={time.time()-t0:.0f}s',flush=True)
lp=val();np.save(D+f'results/hslseq_{a.tag}_valLP.npy',lp)
# test
o=hslcat.load_full('test2');from gocommon import parse
gi=0;cols=[];hds=[];ns=[]
for nm in ('public','private'):
    for r in csv.DictReader(open(D+f'data/tests2/tests/rank_prediction_test_{nm}.csv')):
        n=int(r['num_games']);ns.append(n)
        for i in range(1,n+1):h,_=parse(r[f'sgf_{i}']);cols.append(0 if r[f'color_{i}']=='B' else 1);hds.append(h)
Xt=prep(o,np.array(cols),np.array(hds));mm=Xt[:,:,3*R+3]>0;Xt[:,:,:3*R+2]=np.where(mm[:,:,None],(Xt[:,:,:3*R+2]-mu)/sd,0);Xt=torch.from_numpy(Xt.astype(np.float16)).to(dev)
net.eval();LPs=[];s=0
with torch.no_grad():
    for n in ns:
        e=net.game(Xt[s:s+n].float()).mean(0,keepdim=True);LPs.append(torch.log_softmax(net.head(e),-1)[0].cpu().numpy());s+=n
np.save(D+f'results/hslseq_{a.tag}_testLP.npy',np.array(LPs));print('DONE',flush=True)
