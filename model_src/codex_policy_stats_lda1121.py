"""Train regularized policy metrics with separate development and audit player rows."""
import argparse,csv,hashlib,json,tarfile
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
import torch
from torch import nn
from torch.nn import functional as F

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def z(x):return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-9)
def metric(x,rows):
    order=np.argsort(-x[rows],axis=1)[:,:5]
    return float(((order==rows[:,None])*np.exp(-np.arange(5))[None]).sum(1).mean())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--features',choices=['mean_std','played','combined'],required=True);a=p.parse_args();root=a.root.resolve()
    out=root/'reproduction'/('policy_stats_lda1121_'+a.features);out.mkdir(exist_ok=False)
    tr=root/'reproduction/policy_residual_train1119';ho=root/'reproduction/policy_residual_pilot1119'
    tm=np.load(tr/'sample_metadata.npz');hm=np.load(ho/'sample_metadata.npz')
    keys={'mean_std':['residual_mean','residual_std'],'played':['played_mean'],'combined':['residual_mean','residual_std','played_mean']}[a.features]
    XT=np.load(tr/'policy_residual_features.npz');HT=np.load(ho/'policy_residual_features.npz')
    X=np.concatenate([XT[k] for k in keys],axis=1).astype(np.float64)
    H=np.concatenate([HT[k] for k in keys],axis=1).astype(np.float64)
    neural_H=HT['residual_mean']
    assert np.isfinite(X).all() and np.isfinite(H).all() and not np.intersect1d(tm['player'],hm['player']).size
    players,y=np.unique(tm['player'],return_inverse=True);mean=X.mean(0);scale=np.maximum(X.std(0),1e-4);X=(X-mean)/scale;HS=(H-mean)/scale
    means=np.stack([X[y==i].mean(0) for i in range(len(players))]);residual=X-means[y]
    within=residual.T@residual/max(len(X)-len(players),1);between=np.cov(means,rowvar=False,bias=True)
    rng=np.random.default_rng(11202026);perm=rng.permutation(600);dev=np.sort(perm[:300]);audit=np.sort(perm[300:])
    ids=hm['player'].reshape(600,15);assert all(len(set(row))==1 for row in ids)
    np.savez(out/'selection_split.npz',development_player_ids=ids[dev,0],audit_player_ids=ids[audit,0],training_player_ids=players)
    cached=np.load(root/'results/HO_meta.npz');rowof={int(x):i for i,x in enumerate(cached['idx'])}
    baselines=[norm(np.load(root/'results'/f'HO_{tag}.npy')[[rowof[int(i)] for i in hm['idx']]].astype(np.float32)).reshape(600,15,-1) for tag in ['a2','b1','b2','b3','h4','h5']]
    ck=torch.load(root/'reproduction/policy_metric1119/policy_metric.pt',map_location='cpu',weights_only=False)
    net=nn.Sequential(nn.LayerNorm(len(ck['mean'])),nn.Linear(len(ck['mean']),ck['hidden']),nn.GELU(),nn.Dropout(.15),nn.Linear(ck['hidden'],ck['dimension']));net.load_state_dict(ck['net']);net.eval()
    rawH=torch.tensor(((neural_H-ck['mean'])/ck['scale']).astype(np.float32))
    with torch.inference_mode():M=np.concatenate([F.normalize(net(rawH[i:i+1024]),dim=-1).numpy() for i in range(0,len(rawH),1024)]).reshape(600,15,-1)
    splits=[]
    for seed in range(5):
        rng=np.random.default_rng(71190+seed);sp=np.stack([rng.permutation(15) for _ in range(600)]);ref,q=sp[:,:10],sp[:,10:];base={}
        for k in [1,2,3,5]:
            base[k]=sum(z(norm(v[np.arange(600)[:,None],q[:,:k]].mean(1))@norm(v[np.arange(600)[:,None],ref].mean(1)).T) for v in baselines)/6
            base[k]+=.25*z(norm(M[np.arange(600)[:,None],q[:,:k]].mean(1))@norm(M[np.arange(600)[:,None],ref].mean(1)).T)
        splits.append((ref,q,base))
    def evaluate(E,w,group):
        result={k:[] for k in [1,2,3,5]}
        for ref,q,base in splits:
            prototypes=norm(E[np.arange(600)[:,None],ref].mean(1))
            for k in result:result[k].append(metric(base[k]+w*z(norm(E[np.arange(600)[:,None],q[:,:k]].mean(1))@prototypes.T),group))
        return {str(k):float(np.mean(v)) for k,v in result.items()}
    base_dev={str(k):float(np.mean([metric(b[k],dev) for _,_,b in splits])) for k in [1,2,3,5]}
    base_audit={str(k):float(np.mean([metric(b[k],audit) for _,_,b in splits])) for k in [1,2,3,5]}
    print('BASELINE_DEV',base_dev,flush=True)
    best=None;hist=[];transforms={}
    for shrink in [.1,.5]:
        W=(1-shrink)*within+shrink*np.trace(within)/len(mean)*np.eye(len(mean))
        val,vec=eigh(between,W)
        for dim in [64,128]:
            # Unit generalized discriminants avoid amplifying training identity eigenvalues.
            A=vec[:,-dim:].astype(np.float32);tag=f'lda_s{shrink}_d{dim}';transforms[tag]=A
            E=norm(HS@A).astype(np.float32).reshape(600,15,-1)
            for w in [.1,.25,.5]:
                score=evaluate(E,w,dev);rec={'tag':tag,'shrinkage':shrink,'dimension':dim,'weight':w,'development_by_k':score,'development_mean':float(np.mean(list(score.values())))};hist.append(rec);print('DEV',json.dumps(rec),flush=True)
                if best is None or rec['development_mean']>best['development_mean']:best=rec
    A=transforms[best['tag']];E=norm(HS@A).astype(np.float32).reshape(600,15,-1)
    audited=evaluate(E,best['weight'],audit)
    report={'selected':best,'development_baseline_by_k':base_dev,'audit_baseline_by_k':base_audit,'audit_selected_by_k':audited,'audit_gain':float(np.mean(list(audited.values()))-np.mean(list(base_audit.values()))),'training_players':len(players),'history':hist,'limitations':['Audit identities were used in earlier experiments; newly reserved audit rows are excluded from this experiment setting selection, but are not a pristine holdout.','Pool has 600 identities, with separate 300-query-row development and audit subsets.','Website evaluation pending; all November 20 slots exhausted.']}
    np.savez(out/'policy_lda.npz',mean=mean,scale=scale,projection=A,weight=best['weight'])
    (out/'pretest_validation.json').write_text(json.dumps(report,indent=2))
    print('STATS_LOCAL_AUDIT',json.dumps({k:report[k] for k in ['selected','audit_baseline_by_k','audit_selected_by_k','audit_gain']}),flush=True)
    import time
    for attempt in range(1200):
        if all((root/'reproduction/policy_test_stats1121'/f'{pool}_regression.json').exists() for pool in ['pub2','priv2']):break
        time.sleep(30)
    else:raise RuntimeError('Timed out waiting for stats extraction')
    parent=root/'reproduction/cx38_candidate_pipeline1118';rows=list(csv.reader((parent/'sub/go19_nover.csv').open(newline='')));replace={}
    for pool in ['pub2','priv2']:
        d=np.load(root/'reproduction/policy_test_stats1121'/f'{pool}_features.npz');feat=np.concatenate([d['features' if k=='residual_mean' else k] for k in keys],axis=1);valid=np.isfinite(feat).all(1)&(d['counts']>0);assert valid.all()
        emb=norm(((feat-mean)/scale)@A);prototypes=norm(emb[:6000].reshape(600,10,-1).mean(1));queries=np.stack([emb[6000:][d['query_indices']==i].mean(0) for i in range(400)])
        lda=z(norm(queries)@prototypes.T);learned=np.load(root/'reproduction/policy_sweep1120'/f'learned_{pool}.npy');base=np.load(parent/'results'/f'Sfinal_review_candidate_{pool}.npy');combined=z(base)+.25*learned+best['weight']*lda
        pids=d['player_ids'];assert np.array_equal(pids,np.load(parent/'results'/f'S_{pool}_pids.npy'))
        np.save(out/f'lda_{pool}.npy',lda)
        for qid,order in zip(d['question_ids'],np.argsort(-combined,axis=1)[:,:5]):replace[str(qid)]=[str(qid)]+[str(pids[j]) for j in order]
    candidate=[rows[0]]+[replace.get(row[0],row) for row in rows[1:]];assert len(candidate)==1601 and all(len(row)==6 and len(set(row[1:]))==5 for row in candidate[1:])
    f=out/('go21_stats_'+a.features+'.csv')
    with f.open('w',newline='') as handle:csv.writer(handle).writerows(candidate)
    report.update(candidate_file=str(f),candidate_sha256=sha(f),public_score=None,changed_top1_vs_parent=sum(x[1]!=y[1] for x,y in zip(candidate[1:],rows[1:])))
    (out/'validation.json').write_text(json.dumps(report,indent=2));print('AUDIT',json.dumps(report),flush=True)
    (out/'REPRODUCE.md').write_text('Restore the frozen Nov20 policy best and extracted train/holdout features; restore the six HO embedding files and HO_meta from the parent backup, and learned score arrays from go-policy-sweep-20261120. Run OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python src/codex_policy_lda1120.py --root PROJECT_ROOT. No GPU is required. Output directory must not exist. Model settings are chosen using development rows only; the audit score is evaluated once after selection. Use validation.json to decide whether website evaluation is justified. No automatic competition upload occurs.\n')
    stage=root/'reproduction/auto_backup_1118'/('policy_stats_lda1121_'+a.features);stage.mkdir(exist_ok=False);archive=stage/('policy_stats_lda1121_'+a.features+'.tar.gz')
    with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=str(out.relative_to(root)));t.add(Path(__file__),arcname='src/codex_policy_stats_lda1121.py')
    (stage/'backup_manifest.json').write_text(json.dumps({'release_tag':'go-policy-stats-20261121','source_commit':(root/'reproduction/policy_stats_source_commit1121.txt').read_text().strip(),'assets':[{'path':str(archive),'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}]},indent=2))
    print('BACKUP_QUEUED',flush=True)
if __name__=='__main__':main()
