"""Generalised NB for Task1 (old black-only set + new mixed-colour few-shot set). tokens(m,c,own,flags) -> token ids; vocab offset by own colour."""
import sys,numpy as np,scipy.sparse as sp,time
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/')
H=1<<18;V0=1+12*361+10*361+2*17*17+5*H;V=2*V0
def tokens(m,c,own,flags=(1,1,1,1,0)):
    n=int((m!=65535).sum());mv=m[:n].astype(np.int64);cc=c[:n];ow=np.flatnonzero(cc==own);t=[];off=own*V0
    pos=mv[ow]
    for k,(p,pl) in enumerate(zip(pos,ow)):
        if p==361:t.append(off);continue
        if k<12:t.append(off+1+k*361+p)
        t.append(off+1+12*361+min(pl//20,9)*361+p)
    if flags[0]:
        base=1+12*361+10*361
        for k in range(1,len(ow)):
            a,b=mv[ow[k-1]],mv[ow[k]]
            if a>=361 or b>=361:continue
            dx=max(-8,min(8,b//19-a//19));dy=max(-8,min(8,b%19-a%19));t.append(off+base+(dx+8)*17+(dy+8))
        base+=17*17
        for k in range(len(ow)):
            i=ow[k]
            if i==0:continue
            a,b=mv[i-1],mv[i]
            if a>=361 or b>=361 or cc[i-1]==own:continue
            dx=max(-8,min(8,b//19-a//19));dy=max(-8,min(8,b%19-a%19));t.append(off+base+(dx+8)*17+(dy+8))
    base=1+12*361+10*361+2*17*17;pw=pos
    if flags[1]:
        for m_ in (2,3,4,5,6,8,10,14):
            if len(pw)>=m_:t.append(off+base+hash(tuple(pw[:m_].tolist()))%H)
    base+=H
    if flags[2]:
        for L_ in (4,8,12,20,30):
            if n>=L_:t.append(off+base+hash((own,)+tuple(mv[:L_].tolist()))%H)
    base+=H
    if flags[3]:
        for k in range(1,min(len(pw),60)):t.append(off+base+hash((int(pw[k-1]),int(pw[k])))%H)
    base+=H
    if len(flags)>4 and flags[4]:
        t.append(off+base+hash(('len',n//12))%H);t.append(off+base+hash(('pass',int((pos==361).sum())))%H)
        t.append(off+base+hash(('len2',n//30,int((pos==361).sum())>0))%H)
        for k in range(min(len(pw),6)):t.append(off+base+hash(('first',k,int(pw[k])//19//3,int(pw[k])%19//3))%H)
    base+=H
    if len(flags)>5 and flags[5]:
        for k in range(len(ow)):
            i=ow[k]
            if i>0 and cc[i-1]!=own:t.append(off+base+hash(('rep',int(mv[i-1]),int(mv[i])))%H)
            if k>=2:t.append(off+base+hash(('tri',int(pw[k-2]),int(pw[k-1]),int(pw[k])))%H)
            if i>=2:t.append(off+base+hash(('win',int(mv[i-2]),int(mv[i-1]),int(mv[i])))%H)
    return t
def mat(M,C,idx,own,flags=(1,1,1,1,0)):
    r=[];c=[]
    for i,(g,o) in enumerate(zip(idx,own)):
        t=tokens(M[g],C[g],int(o),flags);r+=[i]*len(t);c+=t
    return sp.csr_matrix((np.ones(len(r),np.float32),(r,c)),shape=(len(idx),V))
def logp(S,alpha):
    gl=S.sum(0)+1;gl=gl/gl.sum();return np.log((S+alpha*gl[None])/(S.sum(1,keepdims=True)+alpha))
def sim_new(flags=(1,1,1,1),alpha=20,seed=0,npl=600,K=10):
    z=np.load(D+'data/train_arr.npz');M,C,col,P=z['M'],z['C'],z['col'],z['P'];rng=np.random.default_rng(seed)
    cnt=np.bincount(P);elig=np.flatnonzero(cnt>=K+5);pl=rng.choice(elig,npl,replace=False);ix={p:np.flatnonzero(P==p) for p in pl}
    cand=[];qry=[]
    for p in pl:
        g=rng.permutation(ix[p]);cand.append(g[:K]);qry.append(g[K:K+5])
    Xc=[np.asarray(mat(M,C,g,col[g],flags).sum(0)) for g in cand];S=np.vstack(Xc).astype(np.float64);LP=logp(S,alpha)
    Q=[mat(M,C,g,col[g],flags) for g in qry];res={}
    for n in (1,2,3,5):
        s=[]
        for _ in range(400):
            j=rng.integers(npl);q=np.asarray(Q[j][:n].sum(0)).ravel();o=np.argsort(-(LP@q))[:5];r=np.flatnonzero(o==j);s.append(np.exp(-r[0]) if len(r) else 0)
        res[n]=np.mean(s)
    print('NEW sim flags',flags,'alpha',alpha,{k:round(v,3) for k,v in res.items()},'ALL %.4f'%np.mean(list(res.values())),flush=True)
if __name__=='__main__':
    for fl in [(1,1,1,1),(0,1,1,1),(1,0,0,0),(0,0,0,1)]:
        sim_new(fl,20)
    for al in (3,60):sim_new((1,1,1,1),al)
