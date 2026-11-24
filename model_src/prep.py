"""CSV -> compact arrays. moves int16 (N,L,3)? use uint8 (N,400) encoding pos=x*19+y (0..360), 361=pass, 255=pad ; col (N,400) 0/1 ; meta."""
import csv,sys,glob,json,numpy as np
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
from gocommon import parse,RANKS,RANK2I
csv.field_size_limit(10**9)
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/')
def enc(mv):
    a=np.full(400,65535,np.uint16);c=np.zeros(400,np.uint8)
    for i,(cc,x,y) in enumerate(mv[:400]):a[i]=361 if x<0 else x*19+y;c[i]=cc
    return a,c
import os;SUB=float(os.environ.get('SUBSAMPLE','1'));RNG=np.random.default_rng(0)
rows=[]
for f in sorted(glob.glob(D+'data/training/train_*.csv')):
    for r in csv.DictReader(open(f)):
        if SUB>=1 or RNG.random()<SUB:rows.append(r)
    print(f,len(rows),flush=True)
N=len(rows);M=np.zeros((N,400),np.uint16);C=np.zeros((N,400),np.uint8);hd=np.zeros(N,np.uint8);y=np.zeros(N,np.uint8);col=np.zeros(N,np.uint8);nm=np.zeros(N,np.uint16);pid=[];gid=[]
for i,r in enumerate(rows):
    h,mv=parse(r['sgf_content']);M[i],C[i]=enc(mv);hd[i]=h;y[i]=RANK2I[r['rank']];col[i]=0 if r['color']=='B' else 1;nm[i]=len(mv);pid.append(r['player_id']);gid.append(r['game_id'])
up={p:i for i,p in enumerate(sorted(set(pid)))};P=np.array([up[p] for p in pid],np.int32)
np.savez(D+'data/train_arr.npz',M=M,C=C,hd=hd,y=y,col=col,nm=nm,P=P)
print('players',len(up),'games',N,'moves mean',nm.mean(),'hd',np.bincount(hd))
# per-rank players, games per player
for k in range(10):
    s=P[y==k];print(RANKS[k],'players',len(set(s)),'games/player',len(s)/len(set(s)),'black frac',(col[y==k]==0).mean())
