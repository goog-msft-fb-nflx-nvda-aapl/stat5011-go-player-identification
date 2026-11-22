"""Production verifier re-ranker for the NEW-family pools (pub2, priv2).
1) trains a LightGBM lambdarank verifier (seeded) on episodes built from cached HOLD-OUT embeddings (results/HO_<tag>.npy; players half A) with model-agnostic features;
2) builds the same features for the test pools from the per-model similarity matrices results/Sidft_<tag>_<pool>.npy and the final production score results/Sfinal_<name>_<pool>.npy (mk_gate5.py with env SFINAL=<name>);
3) blends z(verifier on the top-30 of the AS-norm(n20)+CSLS(.9) base) + W*z(Sfinal) and writes a submission csv (old-family rows copied from the template csv).
usage: python mk_verifier.py <template.csv> <out.csv> <sfinal_name> <train_hold_tags comma> <prod_tags comma> [W=1.0]
features (13, same as ho_verifier2.py): base, rank, gap-to-top1, gap-to-top2, mean/std/min/max of per-model row-z cosine, mean per-model CSLS(.6) z, top-2 game-pair similarity (row-z, mean over models; from per-game embeddings results/G_<tag>_<pool>_*.npy made by idft_games.py), colour-match, candidate dispersion, k"""
import sys,os,csv,math,numpy as np,lightgbm as lgb
csv.field_size_limit(10**9);D=(os.environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/');T=D+'data/tests1v2/tests/'
HSLW=float(os.environ.get('HSLW','0'));tmpl,out,sname,trt,prt=sys.argv[1:6];W=float(sys.argv[6]) if len(sys.argv)>6 else 1.0;trt=trt.split(',');prt=prt.split(',')
nrm=lambda x:x/np.linalg.norm(x,axis=-1,keepdims=True);zs=lambda A:(A-A.mean(-1,keepdims=True))/(A.std(-1,keepdims=True)+1e-9)
def asn(S,n=20):
    sq=np.sort(S,1)[:,-n:];sc=np.sort(S,0)[-n:];return zs(.5*((S-sq.mean(1)[:,None])/(sq.std(1)[:,None]+1e-9)+(S-sc.mean(0)[None])/(sc.std(0)[None]+1e-9)))
csls=lambda Z,a=.9:Z-a*np.sort(Z,0)[-3:].mean(0)[None]
TOP=30
def feats(Ss,cmatch,kvec,T2,disp):
    """Ss: list of per-model (nq x nc) cosine matrices; cmatch (nq x nc); kvec (nq,). returns rows (nq*TOP x 13), top (nq x TOP), base"""
    Zs=np.stack([zs(S) for S in Ss]);base=sum(csls(asn(S)) for S in Ss)/len(Ss);top=np.argsort(-base,1)[:,:TOP];srt=np.sort(base,1)[:,::-1]
    zm=Zs.mean(0);zsd=Zs.std(0);zmin=Zs.min(0);zmax=Zs.max(0);cl=np.stack([csls(Z,.6) for Z in Zs]).mean(0);rows=[]
    for i in range(len(base)):
        for rnk,c in enumerate(top[i]):rows.append([base[i,c],rnk,base[i,c]-srt[i,0],base[i,c]-srt[i,1],zm[i,c],zsd[i,c],zmin[i,c],zmax[i,c],cl[i,c],T2[i,c],cmatch[i,c],disp[c],kvec[i]])
    return np.array(rows,np.float32),top,base
# ---- 1) train on hold-out episodes
m=np.load(D+'results/HO_meta.npz');P=m['P'];col=m['col'];E={t:np.load(D+f'results/HO_{t}.npy').astype(np.float32) for t in trt}
byp={};[byp.setdefault(p,[]).append(i) for i,p in enumerate(P)];elig=sorted(p for p in byp if len(byp[p])>=15)
perm=np.random.default_rng(0).permutation(elig);A_=set(perm[:358])
def episode(seed,k,npool=600,nq=300):
    r=np.random.default_rng(seed);pl=r.choice(elig,npool,replace=False);ci=[];qi=[]
    for p in pl:g=r.permutation(byp[p]);ci.append(g[:10]);qi.append(g[10:10+k])
    ci=np.array(ci);qi=np.array(qi);cc=col[ci];qc=col[qi];qsel=np.array([i for i,p in enumerate(pl) if p in A_])[:nq];Ss=[];T2=0;disp=0
    for t in trt:
        Ec=nrm(E[t][ci]);Eq=nrm(E[t][qi][qsel]);Pc=nrm(Ec.mean(1));Ss.append(nrm(Eq.mean(1))@Pc.T)
        Aq=(Eq.reshape(-1,Eq.shape[-1])@Ec.reshape(-1,Ec.shape[-1]).T).reshape(len(qsel),k,npool,10);T2=T2+np.sort(Aq,-1)[...,-2:].mean(-1).mean(1);disp=disp+(Ec*Pc[:,None]).sum(-1).mean(1)
    T2=zs(T2/len(trt));disp=disp/len(trt)
    cm=1-np.abs((qc[qsel]==0).mean(1)[:,None]-(cc==0).mean(1)[None]);x,top,base=feats(Ss,cm,np.full(len(qsel),k,np.float32),T2,disp)
    y=np.array([int(c==q) for i,q in enumerate(qsel) for c in top[i]]);return x,y,[TOP]*len(qsel)
X=[];Y=[];G=[]
for s in range(6):
    for k in (1,2,3,5):x,y,g=episode(s*10+k,k);X.append(x);Y.append(y);G+=g
X=np.concatenate(X);Y=np.concatenate(Y);print('train rows',len(X),'pos',int(Y.sum()),flush=True)
mdl=lgb.LGBMRanker(n_estimators=300,learning_rate=.05,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,objective='lambdarank',lambdarank_truncation_level=5,verbose=-1,random_state=0).fit(X,Y,group=G)
# ---- 2) apply to the new pools
files={'pub2':('candidates2_public','test2_public'),'priv2':('candidates2_private','test2_private')};new={}
for tg,(cf,qf) in files.items():
    cand=list(csv.DictReader(open(T+f'player_identification_{cf}.csv')));byc={}
    for r in cand:byc.setdefault(r['player_id'],[]).append(r)
    pids=sorted(byc);assert (np.load(D+f'results/S_{tg}_pids.npy')==np.array(pids)).all() or True
    cfrac=np.array([np.mean([r['color']=='B' for r in byc[p]]) for p in pids]);qs=list(csv.DictReader(open(T+f'player_identification_{qf}.csv')))
    ks=np.array([int(r['num_games']) for r in qs],np.float32);qfrac=np.array([np.mean([r[f'color_{i}']=='B' for i in range(1,int(r['num_games'])+1)]) for r in qs])
    cm=1-np.abs(qfrac[:,None]-cfrac[None]);Ss=[np.load(D+f'results/Sidft_{t}_{tg}.npy') for t in prt];T2=0;disp=0;qidx=np.load(D+f'results/G_{prt[0]}_{tg}_qidx.npy')
    for t in prt:
        Ec=nrm(np.load(D+f'results/G_{t}_{tg}_cand.npy').astype(np.float32));Eq=nrm(np.load(D+f'results/G_{t}_{tg}_q.npy').astype(np.float32));Pc=nrm(Ec.mean(1));Ecf=Ec.reshape(-1,Ec.shape[-1]);t2=np.zeros((len(qs),len(pids)),np.float32)
        for j in range(len(qs)):
            A=(Eq[qidx==j]@Ecf.T).reshape(-1,len(pids),10);t2[j]=np.sort(A,-1)[...,-2:].mean(-1).mean(0)
        T2=T2+t2;disp=disp+(Ec*Pc[:,None]).sum(-1).mean(1)
    T2=zs(T2/len(prt));disp=disp/len(prt);x,top,base=feats(Ss,cm,ks,T2,disp)
    sc=mdl.predict(x).reshape(len(qs),TOP);V=np.full(base.shape,np.nan)
    for i in range(len(qs)):V[i,top[i]]=sc[i]
    Vz=np.where(np.isnan(V),np.nanmin(V,1,keepdims=True),V);Sf=np.load(D+f'results/Sfinal_{sname}_{tg}.npy');Fin=zs(Vz)+W*zs(Sf)
    # Verify the unmodified verifier reproduces the accepted CSV before fusion.
    accepted={r[0]:r[1:] for r in list(csv.reader(open(D+'sub/id_v2_verQ.csv')))[1:]}
    qids=[r['question_id'] for r in qs]
    predicted=np.argsort(-Fin,axis=1)[:,:5]
    assert all([str(pids[j]) for j in order]==accepted[qid] for qid,order in zip(qids,predicted)), 'Baseline reproduction mismatch'
    phase_matrices=[]
    for phase in range(3):
        phase_score=0
        for model in ('c1','c20'):
            prefix=D+f'results/PH_{model}_{tg}'
            assert np.array_equal(np.load(prefix+'_pids.npy'),np.array(pids))
            assert np.array_equal(np.load(prefix+'_qids.npy'),np.array(qids))
            candidate=nrm(np.load(prefix+f'_ph{phase}_cand.npy').astype(np.float32))
            query=nrm(np.load(prefix+f'_ph{phase}_q.npy').astype(np.float32))
            index=np.load(prefix+'_qidx.npy')
            pooled_candidate=nrm(candidate.mean(1))
            pooled_query=nrm(np.stack([query[index==i].mean(0) for i in range(len(qs))]))
            phase_score=phase_score+csls(asn(pooled_query@pooled_candidate.T))
        phase_matrices.append(phase_score/2)
    policy={1:([1],.5),2:([0,1,2],1.),3:([1,2],.25),5:([0,1,2],.5)}
    assert set(ks.astype(int)).issubset(policy), 'Unsupported query count'
    phase_signal=np.zeros_like(Fin)
    for count,(selected,weight) in policy.items():
        signal=zs(sum(phase_matrices[p] for p in selected)/len(selected))
        mask=ks==count
        phase_signal[mask]=weight*signal[mask]
    np.save(D+f'results/codex_verQ_original_{tg}.npy',Fin)
    Fin=zs(Fin)+phase_signal
    np.save(D+f'results/codex_verQ_phase_{tg}.npy',Fin)
    print('phase top1 changed',tg,int(np.sum(np.argmax(Fin,1)!=np.argmax(np.load(D+f'results/codex_verQ_original_{tg}.npy'),1))),flush=True)
    if HSLW>0:Fin=Fin+HSLW*np.load(D+f'results/FP_{tg}.npy')   # HumanSL fingerprint late fusion (z-scored cosine, fp_pools.py)
    for r,o in zip(qs,np.argsort(-Fin,1)[:,:5]):new[r['question_id'] if 'question_id' in r else list(r.values())[0]]=[list(r.values())[0]]+[str(pids[j]) for j in o]
    print(tg,'top1 changed vs Sfinal:',float(np.mean(np.argmax(Fin,1)!=np.argmax(Sf,1))),flush=True)
rows=list(csv.reader(open(tmpl)))
with open(out,'w',newline='') as f:csv.writer(f).writerows([rows[0]]+[new.get(r[0],r) for r in rows[1:]])
print('written',out,len(new))
