"""(env HSL_PLIES=350 also merges plies 251-350 from results/hsl4_<name>_s{k}.npy, default 250) Merge first-100-ply HumanSL features (results/hsl2_<name>.npy) with the late plies 100-250 (results/hsl3_<name>_s{0,1,2}.npy, interleaved shards [k::3]) -> (G,250,27) float16."""
import numpy as np,os
NP=int(os.environ.get('HSL_PLIES','250'))
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/')
def load_full(name,nsh=3):
    a=np.load(D+f'results/hsl2_{name}.npy',mmap_mode='r');G=len(a);out=np.full((G,NP,a.shape[2]),np.nan,np.float16);out[:,:100]=a
    out[:,:,-1]=np.where(np.isnan(out[:,:,0]),-1,out[:,:,-1])
    for k in range(nsh):
        l=np.load(D+f'results/hsl3_{name}_s{k}.npy',mmap_mode='r');out[k::nsh,100:100+l.shape[1]]=l
    if NP>250:
        for k in range(nsh):
            l=np.load(D+f'results/hsl4_{name}_s{k}.npy',mmap_mode='r');out[k::nsh,250:250+l.shape[1]]=l
    return out
