"""Player-ID fine-tune of pretrained move LM (init lm3): own-move pooled hidden -> 256-d, ArcFace over train players.
usage: CUDA_VISIBLE_DEVICES=3 python train_idft.py <tag> --init lm3 --epochs 3"""
import sys,time,math,argparse,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/jtan/solardefect/go_20261113/src')
from train_lm import LM,sym_tables,D
ap=argparse.ArgumentParser();ap.add_argument('tag');ap.add_argument('--init',default='lm3');ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--bs',type=int,default=128);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--maxlen',type=int,default=320);ap.add_argument('--s',type=float,default=30);ap.add_argument('--m',type=float,default=0.3);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--allp',type=int,default=0);ap.add_argument('--bag',type=int,default=0);ap.add_argument('--resume',default='')
a=ap.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed)
dev=torch.device('cuda',0);t=torch.arange(32,device=dev,dtype=torch.float32);assert (t*t).sum().item()==10416;print('REAL_TENSOR_PASS',flush=True)
ck=torch.load(D+f'results/lm_{a.init}.pt',map_location='cpu',weights_only=False);la=ck['args']
lm=LM(la['d'],la['layers'],la['heads']);lm.load_state_dict(ck['net'])
with torch.no_grad():lm.cond.weight[10]=lm.cond.weight[:10].mean(0)
z=np.load(D+'data/train_arr.npz');M=z['M'];C=z['C'];hd=z['hd'];col=z['col'];P=z['P']
hold=np.random.default_rng(1234).choice(np.unique(P),800,replace=False);ishold=np.isin(P,hold);tr=np.flatnonzero(~ishold) if not a.allp else np.arange(len(P));va=np.flatnonzero(ishold)
cls={p:i for i,p in enumerate(np.unique(P[tr]))};ytr=np.array([cls.get(p,-1) for p in P]);NC=len(cls)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.lm=lm;d=la['d'];s.proj=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,1024),nn.GELU(),nn.Linear(1024,256));s.W=nn.Parameter(torch.randn(NC,256)*0.05)
    def emb(s,pos,col,tgt):
        B,L=pos.shape;m=s.lm;ply=torch.arange(L,device=pos.device)[None].expand(B,L);cond=torch.full((B,),10,device=pos.device);hdz=torch.zeros(B,dtype=torch.long,device=pos.device)
        return pos,col,tgt,ply,cond
    def forward(s,pos,col,hd,tgt):
        B,L=pos.shape;m=s.lm;ply=torch.arange(L,device=pos.device)[None].expand(B,L);cond=torch.full((B,),10,device=pos.device)
        x=m.pos(pos)+m.col(col)+m.ply(ply)+(m.hd(hd)+m.tgt(tgt)+m.cond(cond))[:,None]
        mask=torch.nn.Transformer.generate_square_subsequent_mask(L,device=pos.device)
        h=m.ln(m.enc(x,mask=mask,is_causal=True));own=((col==tgt[:,None])&(pos!=362)).float().unsqueeze(-1);allm=(pos!=362).float().unsqueeze(-1)
        pooled=torch.cat([(h*own).sum(1)/own.sum(1).clamp(min=1),(h*allm).sum(1)/allm.sum(1).clamp(min=1)],1)
        return F.normalize(s.proj(pooled.float()),dim=-1)
Mg=torch.from_numpy(M[:,:a.maxlen].astype(np.int64)).to(dev);Cg=torch.from_numpy(C[:,:a.maxlen].astype(np.int64)).to(dev);Mg[Mg==65535]=362
hdg=torch.from_numpy(np.minimum(hd,9).astype(np.int64)).to(dev);colg=torch.from_numpy(col.astype(np.int64)).to(dev);yg=torch.from_numpy(ytr).to(dev);SYM=sym_tables(dev)
net=Net().to(dev)
if a.resume:net.load_state_dict(torch.load(D+f'results/idft_{a.resume}.pt',map_location='cpu',weights_only=False)['net']);print('resumed',flush=True)
opt=torch.optim.AdamW([{'params':net.lm.parameters(),'lr':a.lr},{'params':list(net.proj.parameters())+[net.W],'lr':1e-3}],weight_decay=0.05,betas=(0.9,0.98))
steps_per=len(tr)//a.bs;total=steps_per*a.epochs;sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr,1e-3],total_steps=total,pct_start=0.05);trg=torch.from_numpy(tr).to(dev)
cnt=np.bincount(P[va],minlength=P.max()+1);elig=np.array([p for p in hold if cnt[p]>=15]);r2=np.random.default_rng(5);pl=r2.choice(elig,min(600,len(elig)),replace=False);cand=[];qry=[]
for p in pl:
    g=r2.permutation(va[P[va]==p]);cand.append(g[:10]);qry.append(g[10:15])
need=np.concatenate(cand+qry);
@torch.no_grad()
def embed(idx,tta=1):
    net.eval();out=[]
    for s in range(0,len(idx),256):
        ib=torch.from_numpy(idx[s:s+256]).to(dev);acc=0
        for k in range(tta):
            pos=torch.gather(SYM[k].expand(len(ib),-1),1,Mg[ib])
            with torch.autocast('cuda',dtype=torch.bfloat16):acc=acc+net(pos,Cg[ib],hdg[ib],colg[ib]).float()
        out.append(F.normalize(acc,dim=-1).cpu())
    net.train();return torch.cat(out).numpy()
def sim():
    E=embed(need);n=len(pl);Ec=E[:n*10].reshape(n,10,-1);Eq=E[n*10:].reshape(n,5,-1);Pc=Ec.mean(1);Pc/=np.linalg.norm(Pc,axis=1,keepdims=True);res={}
    for k in (1,2,3,5):
        q=Eq[:,:k].mean(1);q/=np.linalg.norm(q,axis=1,keepdims=True);S=q@Pc.T;o=np.argsort(-S,1)[:,:5];res[k]=round(float(np.mean([math.exp(-np.flatnonzero(o[i]==i)[0]) if i in o[i] else 0 for i in range(n)])),4)
    return res,E
t0=time.time();net.train()
for ep in range(a.epochs):
    if a.bag:
        byp={}
        for g in tr:byp.setdefault(P[g],[]).append(g)
        plist=[p for p,v in byp.items() if len(v)>=4];tl=0;K4=4;NP=a.bs//K4
        for s in range(steps_per):
            ps=np.random.choice(len(plist),NP,replace=False);ib_np=np.stack([np.random.choice(byp[plist[i]],K4,replace=False) for i in ps]);ib=torch.from_numpy(ib_np.reshape(-1)).to(dev);k=torch.randint(0,8,(len(ib),),device=dev);pos=torch.gather(SYM[k],1,Mg[ib])
            with torch.autocast('cuda',dtype=torch.bfloat16):e=net(pos,Cg[ib],hdg[ib],colg[ib])
            e=e.float().view(NP,K4,-1);y=yg[torch.from_numpy(ib_np[:,0]).to(dev)];Wn=F.normalize(net.W,dim=-1);loss=0
            for emb_ in (e[:,0],F.normalize(e[:,:2].mean(1),dim=-1),F.normalize(e.mean(1),dim=-1)):
                cosv=emb_@Wn.T;th=torch.acos(cosv.clamp(-1+1e-6,1-1e-6));tgt_cos=torch.cos(th.gather(1,y[:,None])+a.m);lg=cosv.scatter(1,y[:,None],tgt_cos)*a.s;loss=loss+F.cross_entropy(lg,y)/3
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();sched.step();tl+=loss.item()
            if s%1000==0:print(f'ep{ep} step{s}/{steps_per} loss {tl/(s+1):.4f} t={time.time()-t0:.0f}s',flush=True)

    else:
        perm=trg[torch.randperm(len(trg),device=dev)];tl=0
        for s in range(steps_per):
            ib=perm[s*a.bs:(s+1)*a.bs];k=torch.randint(0,8,(len(ib),),device=dev);pos=torch.gather(SYM[k],1,Mg[ib])
            with torch.autocast('cuda',dtype=torch.bfloat16):e=net(pos,Cg[ib],hdg[ib],colg[ib])
            cosv=e.float()@F.normalize(net.W,dim=-1).T;y=yg[ib];th=torch.acos(cosv.clamp(-1+1e-6,1-1e-6));tgt_cos=torch.cos(th.gather(1,y[:,None])+a.m);lg=cosv.scatter(1,y[:,None],tgt_cos)*a.s;loss=F.cross_entropy(lg,y)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();sched.step();tl+=loss.item()
            if s%1000==0:print(f'ep{ep} step{s}/{steps_per} loss {tl/(s+1):.4f} t={time.time()-t0:.0f}s',flush=True)
    r,E=sim();print(f'EPOCH {ep} loss {tl/steps_per:.4f} NEW-like sim {r} t={time.time()-t0:.0f}s',flush=True)
    torch.save({'net':net.state_dict(),'args':vars(a)},D+f'results/idft_{a.tag}.pt')
print('DONE',flush=True)
