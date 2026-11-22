"""Importable model/data helpers matching train_seq.py checkpoints."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
D='/home/jtan/solardefect/go_20261113/'
class Net(nn.Module):
    def __init__(s,nplayers,a):
        super().__init__();d=a['d'];s.a=a
        s.pos=nn.Embedding(363,d);s.col=nn.Embedding(3,d);s.ply=nn.Embedding(512,d);s.hd=nn.Embedding(10,d);s.tgt=nn.Embedding(2,d)
        L=nn.TransformerEncoderLayer(d,a['heads'],d*4,0.1,batch_first=True,norm_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(L,a['layers']);s.ln=nn.LayerNorm(d)
        s.rank=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,10));s.emb=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,a['emb']))
        s.W=nn.Parameter(torch.randn(nplayers,a['emb'])*0.02)
    def forward(s,pos,col,hd,tgt,mask):
        B,L=pos.shape;ply=torch.arange(L,device=pos.device)[None].expand(B,L)
        x=s.pos(pos)+s.col(col)+s.ply(ply)+(s.hd(hd)+s.tgt(tgt))[:,None]
        h=s.ln(s.enc(x,src_key_padding_mask=~mask));own=(col==tgt[:,None])&mask;w=own.float().unsqueeze(-1)
        return (h*w).sum(1)/w.sum(1).clamp(min=1)
    def heads(s,p):return s.rank(p),F.normalize(s.emb(p),dim=-1)
def sym_tables(dev):
    xs,ys=np.meshgrid(np.arange(19),np.arange(19),indexing='ij')
    def tab(f):
        X,Y=f(xs,ys);T=np.full(363,362);T[:361]=(X*19+Y).ravel();T[361]=361;return torch.from_numpy(T).to(dev)
    fs=[lambda x,y:(x,y),lambda x,y:(18-x,y),lambda x,y:(x,18-y),lambda x,y:(18-x,18-y),lambda x,y:(y,x),lambda x,y:(18-y,x),lambda x,y:(y,18-x),lambda x,y:(18-y,18-x)]
    return torch.stack([tab(f) for f in fs])
def load_ckpt(path,dev):
    ck=torch.load(path,map_location='cpu',weights_only=False);a=ck['args'];n=ck['net']['W'].shape[0]
    net=Net(n,a);net.load_state_dict(ck['net']);return net.to(dev).eval(),a,ck
def enc_sgfs(sgfs,maxlen=320):
    """list of sgf strings -> pos(N,L) int64 (pad 362), col(N,L), hd(N)"""
    import sys;sys.path.insert(0,D+'src');from gocommon import parse
    N=len(sgfs);pos=np.full((N,maxlen),362,np.int64);col=np.zeros((N,maxlen),np.int64);hd=np.zeros(N,np.int64)
    for i,s in enumerate(sgfs):
        h,mv=parse(s);hd[i]=min(h,9)
        for j,(c,x,y) in enumerate(mv[:maxlen]):pos[i,j]=361 if x<0 else x*19+y;col[i,j]=c
    return pos,col,hd
@torch.no_grad()
def run(net,pos,col,hd,tgt,sym,dev,tta=1,bs=512):
    """-> rank logprobs (N,10), emb (N,E) averaged over tta symmetries"""
    R=[];E=[]
    for s in range(0,len(pos),bs):
        p=torch.from_numpy(pos[s:s+bs]).to(dev);c=torch.from_numpy(col[s:s+bs]).to(dev);h=torch.from_numpy(hd[s:s+bs]).to(dev);t=torch.from_numpy(tgt[s:s+bs]).to(dev)
        r_acc=0;e_acc=0
        for k in range(tta):
            pp=torch.gather(sym[k].expand(len(p),-1),1,p) if k>0 or tta>1 else p
            m=pp!=362
            with torch.autocast('cuda',dtype=torch.bfloat16):pooled=net(pp,c,h,t,m)
            r,e=net.heads(pooled.float());r_acc=r_acc+F.log_softmax(r,-1);e_acc=e_acc+e
        R.append((r_acc/tta).cpu());E.append(F.normalize(e_acc,dim=-1).cpu())
    return torch.cat(R).numpy(),torch.cat(E).numpy()
