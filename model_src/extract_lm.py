"""Frozen LM (lm2) pooled hidden states per game: [own-colour mean, all-moves mean] -> (N,2d) fp16 saved as results/rank_svd{2d}_Z.npy for reuse by id_mlp.py (ZNC=2d)."""
import sys,time,numpy as np,torch
sys.path.insert(0,'/home/jtan/solardefect/go_20261113/src')
from train_lm import LM,sym_tables,D
tag=sys.argv[1];dev=torch.device('cuda',0);t=torch.arange(32,device=dev,dtype=torch.float32);assert (t*t).sum().item()==10416;print('REAL_TENSOR_PASS',flush=True)
ck=torch.load(D+f'results/lm_{tag}.pt',map_location='cpu',weights_only=False);la=ck['args'];net=LM(la['d'],la['layers'],la['heads']);net.load_state_dict(ck['net']);net=net.to(dev).eval()
with torch.no_grad():net.cond.weight[10]=net.cond.weight[:10].mean(0)
z=np.load(D+'data/train_arr.npz');M,C,hd,col=z['M'],z['C'],z['hd'],z['col'];N=len(M);d=la['d'];out=np.zeros((N,2*d),np.float16);t0=time.time();L=320
with torch.no_grad():
    for s in range(0,N,512):
        pos=torch.from_numpy(M[s:s+512,:L].astype(np.int64)).to(dev);pos[pos==65535]=362;cc=torch.from_numpy(C[s:s+512,:L].astype(np.int64)).to(dev);h_=torch.from_numpy(np.minimum(hd[s:s+512],9).astype(np.int64)).to(dev);tg=torch.from_numpy(col[s:s+512].astype(np.int64)).to(dev)
        B=pos.shape[0];ply=torch.arange(L,device=dev)[None].expand(B,L);cond=torch.full((B,),10,device=dev)
        x=net.pos(pos)+net.col(cc)+net.ply(ply)+(net.hd(h_)+net.tgt(tg)+net.cond(cond))[:,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):h=net.ln(net.enc(x,mask=torch.nn.Transformer.generate_square_subsequent_mask(L,device=dev),is_causal=True))
        h=h.float();own=((cc==tg[:,None])&(pos!=362)).float().unsqueeze(-1);allm=(pos!=362).float().unsqueeze(-1)
        out[s:s+B]=torch.cat([(h*own).sum(1)/own.sum(1).clamp(min=1),(h*allm).sum(1)/allm.sum(1).clamp(min=1)],1).cpu().numpy().astype(np.float16)
        if (s//512)%200==0:print(s,N,time.time()-t0,flush=True)
np.save(D+f'results/rank_svd{2*d}_Z.npy',out);print('done',2*d,time.time()-t0,flush=True)
