"""Rank predictions from fine-tuned models (train_ft.py checkpoints): sum of per-game log-probs (TTA 8 syms, optional multi-model average), expected-score decision. usage: rank_predict_ft.py out.csv tag1[,tag2..] [tau]"""
import sys,csv,math,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/jtan/solardefect/go_20261113/src')
from train_lm import LM,sym_tables,D
from gocommon import parse,RANKS
csv.field_size_limit(10**9);out=None;tags=['ft6_sord8','ft7_sord8','ft8_sord8','ft9_sord8'];dev=torch.device('cuda',0);sym=sym_tables(dev)
class Net(nn.Module):
    def __init__(s,la,loss,pool='own'):
        super().__init__();s.lm=LM(la['d'],la['layers'],la['heads']);d=la['d'];s.loss=loss;s.pool=pool;di=d if pool=='own' else 2*d
        s.head=nn.Sequential(nn.LayerNorm(di),nn.Linear(di,d),nn.GELU(),nn.Linear(d,1 if loss=='coral' else 10))
        if loss=='coral':s.bias=nn.Parameter(torch.zeros(9))
    def forward(s,pos,col,hd,tgt):
        B,L=pos.shape;m=s.lm;ply=torch.arange(L,device=pos.device)[None].expand(B,L);cond=torch.full((B,),10,device=pos.device)
        x=m.pos(pos)+m.col(col)+m.ply(ply)+(m.hd(hd)+m.tgt(tgt)+m.cond(cond))[:,None]
        mask=torch.nn.Transformer.generate_square_subsequent_mask(L,device=pos.device);h=m.ln(m.enc(x,mask=mask,is_causal=True))
        own=((col==tgt[:,None])&(pos!=362)).float().unsqueeze(-1);pooled=(h*own).sum(1)/own.sum(1).clamp(min=1)
        if s.pool=='both':
            allm=(pos!=362).float().unsqueeze(-1);pooled=torch.cat([pooled,(h*allm).sum(1)/allm.sum(1).clamp(min=1)],1)
        return s.head(pooled.float())
    def logprobs(s,o):
        if s.loss=='coral':
            cum=torch.sigmoid(o+s.bias[None]);p=torch.cat([1-cum[:,:1],cum[:,:-1]-cum[:,1:],cum[:,-1:]],1).clamp(min=1e-6);return torch.log(p/p.sum(1,keepdim=True))
        return F.log_softmax(o,-1)
def load(tag):
    ck=torch.load(D+f'results/ft_{tag}.pt',map_location='cpu',weights_only=False);a=ck['args']
    lck=torch.load(D+f"results/lm_{a['init']}.pt",map_location='cpu',weights_only=False);la=lck['args'];net=Net(la,a['loss'],a.get('pool','own'));net.load_state_dict(ck['net']);return net.to(dev).eval()
nets=[load(t) for t in tags]
def game_lp(sgfs,colours):
    N=len(sgfs);pos=np.full((N,320),362,np.int64);col=np.zeros((N,320),np.int64);hd=np.zeros(N,np.int64);tg=np.array([0 if c=='B' else 1 for c in colours],np.int64)
    for i,s in enumerate(sgfs):
        h,mv=parse(s);hd[i]=min(h,9)
        for j,(c,x,y) in enumerate(mv[:320]):pos[i,j]=361 if x<0 else x*19+y;col[i,j]=c
    P=torch.from_numpy(pos).to(dev);Cc=torch.from_numpy(col).to(dev);H=torch.from_numpy(hd).to(dev);T=torch.from_numpy(tg).to(dev);acc=0
    with torch.no_grad():
        for net in nets:
            for k in range(8):
                pp=torch.gather(sym[k].expand(N,-1),1,P)
                with torch.autocast('cuda',dtype=torch.bfloat16):o=net(pp,Cc,H,T)
                acc=acc+net.logprobs(o.float())
    return (acc/(8*len(nets))).cpu().numpy()

T1=D+'data/tests1v2/tests/'
for tag,cf,qf in (('pub2','candidates2_public','test2_public'),('priv2','candidates2_private','test2_private')):
    cand=list(csv.DictReader(open(T1+f'player_identification_{cf}.csv')));byp={}
    for r in cand:byp.setdefault(r['player_id'],[]).append(r)
    pids=sorted(byp);order=[r for p in pids for r in byp[p]];lp=np.concatenate([game_lp([r['sgf_content'] for r in order[s:s+256]],[r['color'] for r in order[s:s+256]]) for s in range(0,len(order),256)])
    own=np.array([i for i,p in enumerate(pids) for _ in byp[p]]);LC=np.stack([lp[own==i].sum(0) for i in range(len(pids))]);np.save(D+f'results/LPc_codex1118_{tag}.npy',LC)
    LQ=[]
    for r in csv.DictReader(open(T1+f'player_identification_{qf}.csv')):
        n=int(r['num_games']);LQ.append(game_lp([r[f'sgf_{i}'] for i in range(1,n+1)],[r[f'color_{i}'] for i in range(1,n+1)]).sum(0))
    np.save(D+f'results/LPq_codex1118_{tag}.npy',np.stack(LQ));print(tag,'done',flush=True)
