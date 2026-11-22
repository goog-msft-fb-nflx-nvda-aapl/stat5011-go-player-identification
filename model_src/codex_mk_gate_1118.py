import sys,csv,numpy as np
csv.field_size_limit(10**9);D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/');T=D+"data/tests1v2/tests/";base=sys.argv[1];out=sys.argv[2];lam_new=float(sys.argv[3]);lam_old=float(sys.argv[4]);sk=float(sys.argv[5]);qT=float(sys.argv[6])
CA=float(sys.argv[10]) if len(sys.argv)>10 else 0.0;GB=float(sys.argv[11]) if len(sys.argv)>11 else 0.0;GT=float(sys.argv[12]) if len(sys.argv)>12 else 0.5;QT=sys.argv[13].split(',') if len(sys.argv)>13 else [];GK=3;HU=int(sys.argv[14]) if len(sys.argv)>14 else 0
def qgraph(S,tg):
    Q=np.concatenate([np.load(D+f'results/Qemb_{t}_{tg}.npy') for t in QT],1);Q=Q/np.linalg.norm(Q,axis=1,keepdims=True);A=Q@Q.T;np.fill_diagonal(A,-1);An=np.zeros_like(A);idx=np.argsort(-A,1)[:,:GK]
    for i in range(len(A)):
        for j in idx[i]:
            if A[i,j]>GT:An[i,j]=A[i,j];An[j,i]=A[i,j]
    An=An/(An.sum(1,keepdims=True)+1e-9);return S+GB*(An@S)
itag=sys.argv[7];wn=float(sys.argv[8]);wo=float(sys.argv[9])
files={"pub2":"test2_public","priv2":"test2_private","pub":"test_public","priv":"test_private"}
def sm(L,T):
    L=L/T[:,None] if np.ndim(T) else L/T;L=L-L.max(1,keepdims=True);p=np.exp(L);return p/p.sum(1,keepdims=True)
zs2=lambda A:(A-A.mean(1,keepdims=True))/(A.std(1,keepdims=True)+1e-9)
new={}
for tg in ("pub2","priv2","pub","priv"):
    old=tg in("pub","priv");lam=lam_old if old else lam_new
    S=np.load(D+f"results/S_{tg}.npy");pids=np.load(D+f"results/S_{tg}_pids.npy");S=zs2(S)+(wo if old else wn)*zs2(np.load(D+f"results/Sidft_{itag}_{tg}.npy"));rows=list(csv.DictReader(open(T+f"player_identification_{files[tg]}.csv")));q=[r["question_id"] for r in rows];nq=np.array([int(r["num_games"]) for r in rows],float)
    LC=np.load(D+(f"results/LPc_{tg}.npy" if old else f"results/LPc_codex1118_{tg}.npy"));LQ=np.load(D+(f"results/LPq_{tg}.npy" if old else f"results/LPq_codex1118_{tg}.npy"));nc=100 if old else 10
    gate=np.log(sm(LQ,np.maximum(1.0,nq/5)*qT if old else np.ones(len(q))*qT)@sm(LC,3.0*nc/10)[:len(pids)].T+1e-3);S=(S-CA*np.sort(S,0)[-3:].mean(0)[None] if (not old and CA>0) else S);S=(qgraph(S,tg) if (not old and GB>0) else S)+lam*zs2(gate)
    if old and sk>0:
        L=S/sk;L=L-L.max()
        for _ in range(30):L=L-np.logaddexp.reduce(L,axis=1,keepdims=True);L=L-np.logaddexp.reduce(L,axis=0,keepdims=True)
        S=L
    if __import__('os').environ.get('SFINAL'):np.save(D+f"results/Sfinal_{__import__('os').environ['SFINAL']}_{tg}.npy",S)
    ordr=np.argsort(-S,1)[:,:5]
    if old and HU>0 and S.shape[0]==S.shape[1]:
        from scipy.optimize import linear_sum_assignment
        ri,ci_=linear_sum_assignment(-S);asg=dict(zip(ri,ci_));ordr=ordr.copy();ch=0
        for i in range(S.shape[0]):
            a_=asg[i];top=list(np.argsort(-S[i])[:HU+2])
            if a_ in top[:HU+2] and a_!=ordr[i,0]:
                rest=[c for c in np.argsort(-S[i])[:6] if c!=a_][:4];ordr[i]=[a_]+rest;ch+=1
        print(tg,'hungarian changed top-1 of',ch,'of',S.shape[0],flush=True)
    for qq,o in zip(q,ordr):new[qq]=[qq]+[str(pids[i]) for i in o]
rows=list(csv.reader(open(base)))
with open(out,"w",newline="") as f:csv.writer(f).writerows([rows[0]]+[new.get(r[0],r) for r in rows[1:]])
print(len(new))
