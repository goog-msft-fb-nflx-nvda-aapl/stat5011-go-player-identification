"""Rank fine-tune with SET-level training: bag of n games of one player -> mean of per-game pooled hidden -> head -> SORD loss.
usage: CUDA_VISIBLE_DEVICES=k python train_ftbag.py <tag> --init lm3 --epochs 3"""
import sys,time,math,argparse,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
from train_lm import LM,sym_tables,D
ap=argparse.ArgumentParser();ap.add_argument('tag');ap.add_argument('--init',default='lm3');ap.add_argument('--tau',type=float,default=1.0);ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--np_',type=int,default=16);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--maxlen',type=int,default=320);ap.add_argument('--nmax',type=int,default=12);ap.add_argument('--steps',type=int,default=7000)
a=ap.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed)
dev=torch.device('cuda',0);t=torch.arange(32,device=dev,dtype=torch.float32);assert (t*t).sum().item()==10416;print('REAL_TENSOR_PASS',flush=True)
ck=torch.load(D+f'results/lm_{a.init}.pt',map_location='cpu',weights_only=False);la=ck['args'];hold=ck['hold'];d=la['d']
lm=LM(la['d'],la['layers'],la['heads']);lm.load_state_dict(ck['net'])
with torch.no_grad():lm.cond.weight[10]=lm.cond.weight[:10].mean(0)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.lm=lm;s.head=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,10))
    def game(s,pos,col,hd,tgt):
        B,L=pos.shape;m=s.lm;ply=torch.arange(L,device=pos.device)[None].expand(B,L);cond=torch.full((B,),10,device=pos.device)
        x=m.pos(pos)+m.col(col)+m.ply(ply)+(m.hd(hd)+m.tgt(tgt)+m.cond(cond))[:,None];mask=torch.nn.Transformer.generate_square_subsequent_mask(L,device=pos.device)
        h=m.ln(m.enc(x,mask=mask,is_causal=True)).float();own=((col==tgt[:,None])&(pos!=362)).float().unsqueeze(-1);allm=(pos!=362).float().unsqueeze(-1)
        return torch.cat([(h*own).sum(1)/own.sum(1).clamp(min=1),(h*allm).sum(1)/allm.sum(1).clamp(min=1)],1)
z=np.load(D+'data/train_arr.npz');M=z['M'];C=z['C'];hd=z['hd'];y=z['y'];col=z['col'];P=z['P']
ishold=np.isin(P,hold);tr=np.flatnonzero(~ishold);va=np.flatnonzero(ishold)
Mg=torch.from_numpy(M[:,:a.maxlen].astype(np.int64)).to(dev);Cg=torch.from_numpy(C[:,:a.maxlen].astype(np.int64)).to(dev);Mg[Mg==65535]=362
hdg=torch.from_numpy(np.minimum(hd,9).astype(np.int64)).to(dev);colg=torch.from_numpy(col.astype(np.int64)).to(dev);SYM=sym_tables(dev)
net=Net().to(dev);opt=torch.optim.AdamW([{'params':net.lm.parameters(),'lr':a.lr},{'params':net.head.parameters(),'lr':1e-3}],weight_decay=0.05,betas=(0.9,0.98))
total=a.steps*a.epochs;sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr,1e-3],total_steps=total,pct_start=0.05)
def byplayer(idx):
    b={}
    for g in idx:b.setdefault(P[g],[]).append(g)
    return {p:np.array(v) for p,v in b.items() if len(v)>=5}
Btr=byplayer(tr);Bva=byplayer(va);ptr=list(Btr);pva=list(Bva);yp=lambda B,p:int(y[B[p][0]])
def qs(lp,t):
    p=np.exp(lp-lp.max());p/=p.sum();sc=p.copy();sc[1:]+=math.exp(-1)*p[:-1];sc[:-1]+=math.exp(-1)*p[1:];k=sc.argmax();return 1. if k==t else (math.exp(-1) if abs(k-t)==1 else 0.)
@torch.no_grad()
def evaluate():
    net.eval();r=np.random.default_rng(3);res=[];
    for n in (5,10,15,20):
        for _ in range(8):
            ps=r.choice(len(pva),100,replace=False);ib=np.concatenate([r.choice(Bva[pva[i]],n,replace=len(Bva[pva[i]])<n) for i in ps]);emb=[]
            for s in range(0,len(ib),256):
                j=torch.from_numpy(ib[s:s+256]).to(dev)
                with torch.autocast('cuda',dtype=torch.bfloat16):emb.append(net.game(torch.gather(SYM[0].expand(len(j),-1),1,Mg[j]),Cg[j],hdg[j],colg[j]))
            E=torch.cat(emb).view(100,n,-1).mean(1);lp=F.log_softmax(net.head(E),-1).cpu().numpy()
            for k,i in enumerate(ps):res.append(qs(lp[k],yp(Bva,pva[i])))
    net.train();return float(np.mean(res))
t0=time.time();net.train()
for ep in range(a.epochs):
    tl=0
    for s in range(a.steps):
        ps=np.random.choice(len(ptr),a.np_,replace=False);n=np.random.randint(5,a.nmax+1);ib=np.concatenate([np.random.choice(Btr[ptr[i]],n,replace=len(Btr[ptr[i]])<n) for i in ps]);yt=torch.tensor([yp(Btr,ptr[i]) for i in ps],device=dev)
        j=torch.from_numpy(ib).to(dev);k=torch.randint(0,8,(len(j),),device=dev)
        with torch.autocast('cuda',dtype=torch.bfloat16):g=net.game(torch.gather(SYM[k],1,Mg[j]),Cg[j],hdg[j],colg[j])
        E=g.view(a.np_,n,-1).mean(1);o=net.head(E).float();r=torch.arange(10,device=dev)[None].float();tg=F.softmax(-(r-yt[:,None].float()).abs()/a.tau,-1);loss=-(tg*F.log_softmax(o,-1)).sum(1).mean()
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();sched.step();tl+=loss.item()
        if s%1000==0:print(f'ep{ep} step{s} loss {tl/(s+1):.4f} t={time.time()-t0:.0f}s',flush=True)
    print(f'EPOCH {ep} loss {tl/a.steps:.4f} set_sim {evaluate():.4f} t={time.time()-t0:.0f}s',flush=True)
    torch.save({'net':net.state_dict(),'args':vars(a)},D+f'results/ftbag_{a.tag}.pt')
print('DONE',flush=True)
