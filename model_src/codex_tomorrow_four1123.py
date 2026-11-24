"""Prepare and independently replay four fixed expanded-policy hypotheses; no uploads."""
import argparse,csv,hashlib,json,tempfile,tarfile
from pathlib import Path
import numpy as np

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def z(x):return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-9)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gaussian(ref,q):
    mu=ref.mean(1);v=ref.var(1);v=.8*v.mean(0,keepdims=True)+.2*v+1e-5;inv=1/v;qq=q.mean(1)
    dist=(qq*qq)@inv.T-2*qq@(mu*inv).T+(mu*mu*inv).sum(1)[None]
    return -.5*dist/(1/q.shape[1]+1/ref.shape[1])-.5*np.log(v).sum(1)[None]
def metric(s,rows):
    order=np.argsort(-s[rows],axis=1)[:,:5]
    return float(((order==rows[:,None])*np.exp(-np.arange(5))[None]).sum(1).mean())
CONFIGS=[
 dict(name='small',file='g23_small.csv',cosine=.10,old_gaussian=.10,new_gaussian=0.,hypothesis='Conservative expanded cosine addition to the verified best LM4 combination'),
 dict(name='full',file='g23_full.csv',cosine=.25,old_gaussian=.10,new_gaussian=0.,hypothesis='Transfer the expanded metric selected locally to the best LM4 combination'),
 dict(name='gadd',file='g23_gadd.csv',cosine=0.,old_gaussian=.10,new_gaussian=.10,hypothesis='Add complementary Gaussian matching learned from 6000 identities'),
 dict(name='greplace',file='g23_greplace.csv',cosine=0.,old_gaussian=0.,new_gaussian=.10,hypothesis='Replace the old 1000-player Gaussian representation with the expanded one'),
]
def test_candidates(root,out):
    parent=root/'reproduction/cx38_candidate_pipeline1118';expanded=root/'reproduction/policy_expanded_lda1122'
    ckpath=expanded/'policy_lda.npz';ck=np.load(ckpath);mean,scale,A=ck['mean'],ck['scale'],ck['projection']
    rows=list(csv.reader((parent/'sub/go19_nover.csv').open(newline='')))
    bestpath=root/'reproduction/best1122/g22_cx40.csv';best=list(csv.reader(bestpath.open(newline='')))
    assert sha(bestpath)=='123a07fc353790bc983a76862d10db96e03399df4160a3e7af44deb293b61fdb'
    replacements={c['name']:{} for c in CONFIGS};controls={'original':{},'accepted':{}};hashes={str(ckpath.relative_to(root)):sha(ckpath)}
    for pool in ['pub2','priv2']:
        dp=root/'reproduction/policy_test1119'/f'{pool}_features.npz';d=np.load(dp);feat=d['features'];assert np.isfinite(feat).all() and (d['counts']>0).all()
        emb=norm(((feat-mean)/scale)@A);refs=emb[:6000].reshape(600,10,-1);qsets=[emb[6000:][d['query_indices']==i] for i in range(400)]
        cosine=z(norm(np.stack([q.mean(0) for q in qsets]))@norm(refs.mean(1)).T)
        newg=z(np.concatenate([gaussian(refs,q[None]) for q in qsets],axis=0))
        paths={'parent':parent/'results'/f'Sfinal_review_candidate_{pool}.npy','lm4':root/'reproduction/experiments_1119/runs/cx40_lm4_noaug_all/results'/f'Sfinal_newmember_candidate_{pool}.npy','mlp':root/'reproduction/policy_sweep1120'/f'learned_{pool}.npy','oldg':root/'reproduction/policy_setmatch1121'/f'lda_{pool}.npy'}
        scores={k:np.load(p) for k,p in paths.items()};pids=d['player_ids'];assert np.array_equal(pids,np.load(parent/'results'/f'S_{pool}_pids.npy'))
        for p in [dp,*paths.values()]:hashes[str(p.relative_to(root))]=sha(p)
        np.save(out/f'expanded_cosine_{pool}.npy',cosine);np.save(out/f'expanded_gaussian_{pool}.npy',newg)
        combined={c['name']:z(scores['lm4'])+.25*scores['mlp']+c['cosine']*cosine+c['old_gaussian']*scores['oldg']+c['new_gaussian']*newg for c in CONFIGS}
        combined['original']=z(scores['parent'])+.25*scores['mlp']+.1*scores['oldg']+.25*cosine
        combined['accepted']=z(scores['lm4'])+.25*scores['mlp']+.1*scores['oldg']
        for name,s in combined.items():
            assert s.shape==(400,600) and np.isfinite(s).all()
            target=replacements[name] if name in replacements else controls[name]
            for q,order in zip(d['question_ids'],np.argsort(-s,axis=1)[:,:5]):target[str(q)]=[str(q)]+[str(pids[i]) for i in order]
    records=[];seen=set()
    for config in [dict(name='original',file='go23_expand.csv'),dict(name='accepted',file='accepted_control.csv'),*CONFIGS]:
        name=config['name'];rep=replacements[name] if name in replacements else controls[name]
        output=[rows[0]]+[rep.get(row[0],row) for row in rows[1:]];f=out/config['file']
        assert len(output)==1601 and [r[0] for r in output]==[r[0] for r in rows]
        assert all(len(r)==6 and len(set(r[1:]))==5 for r in output[1:])
        assert all(a==b for a,b in zip(output[1:],best[1:]) if a[0].startswith(('pub_q_','priv_q_')))
        with f.open('w',newline='') as handle:csv.writer(handle).writerows(output)
        h=sha(f);assert h not in seen;seen.add(h)
        if name=='original':assert h=='9230926b6adcb238a8a56b056a3353445d8e1b60f9f086cd780d9cca726e8b88'
        elif name=='accepted':assert h=='123a07fc353790bc983a76862d10db96e03399df4160a3e7af44deb293b61fdb'
        if name!='accepted':records.append(dict(config,sha256=h,public_score=None,changed_top1_vs_best=sum(a[1]!=b[1] for a,b in zip(output[1:],best[1:]))))
    return records,hashes

def validate(root):
    import torch
    from torch import nn
    from torch.nn import functional as F
    hroot=root/'reproduction/policy_residual_pilot1119';meta=np.load(hroot/'sample_metadata.npz');H=np.load(hroot/'policy_residual_features.npz')['residual_mean'].astype(np.float64)
    new=np.load(root/'reproduction/policy_expanded_lda1122/policy_lda.npz');old=np.load(root/'reproduction/policy_setmatch1121/policy_lda.npz')
    E=norm(((H-new['mean'])/new['scale'])@new['projection']).astype(np.float32).reshape(600,15,-1);O=norm(((H-old['mean'])/old['scale'])@old['projection']).reshape(600,15,-1)
    cached=np.load(root/'results/HO_meta.npz');rowof={int(i):j for j,i in enumerate(cached['idx'])}
    base=[norm(np.load(root/'results'/f'HO_{tag}.npy')[[rowof[int(i)] for i in meta['idx']]].astype(np.float32)).reshape(600,15,-1) for tag in ['a2','b1','b2','b3','h4','h5']]
    ck=torch.load(root/'reproduction/policy_metric1119/policy_metric.pt',map_location='cpu',weights_only=False)
    net=nn.Sequential(nn.LayerNorm(len(ck['mean'])),nn.Linear(len(ck['mean']),ck['hidden']),nn.GELU(),nn.Dropout(.15),nn.Linear(ck['hidden'],ck['dimension']));net.load_state_dict(ck['net']);net.eval();X=torch.tensor(((H-ck['mean'])/ck['scale']).astype(np.float32))
    with torch.inference_mode():M=np.concatenate([F.normalize(net(X[i:i+1024]),dim=-1).numpy() for i in range(0,len(X),1024)]).reshape(600,15,-1)
    perm=np.random.default_rng(11202026).permutation(600);groups={'development':np.sort(perm[:300]),'audit':np.sort(perm[300:])};axis=np.arange(600)[:,None]
    results={name:{group:{str(k):[] for k in [1,2,3,5]} for group in groups} for name in ['baseline',*[c['name'] for c in CONFIGS]]}
    for seed in range(5):
        rng=np.random.default_rng(71190+seed);sp=np.stack([rng.permutation(15) for _ in range(600)]);ref,q=sp[:,:10],sp[:,10:]
        for k in [1,2,3,5]:
            original=sum(z(norm(v[axis,q[:,:k]].mean(1))@norm(v[axis,ref].mean(1)).T) for v in base)/6
            original+=.25*z(norm(M[axis,q[:,:k]].mean(1))@norm(M[axis,ref].mean(1)).T)
            oldg=z(gaussian(O[axis,ref],O[axis,q[:,:k]]));newg=z(gaussian(E[axis,ref],E[axis,q[:,:k]]));cos=z(norm(E[axis,q[:,:k]].mean(1))@norm(E[axis,ref].mean(1)).T)
            baseline=original.copy();baseline+=.1*oldg;scores={'baseline':baseline}
            for c in CONFIGS:
                b=original.copy();b+=c['old_gaussian']*oldg
                scores[c['name']]=b+c['cosine']*cos+c['new_gaussian']*newg
            for name,s in scores.items():
                for group,rr in groups.items():results[name][group][str(k)].append(metric(s,rr))
    report={name:{group:{'by_query_games':{k:float(np.mean(v)) for k,v in data.items()},'mean':float(np.mean([np.mean(v) for v in data.values()]))} for group,data in groupscores.items()} for name,groupscores in results.items()}
    expected=json.loads((root/'reproduction/policy_expanded_lda1122/validation.json').read_text())
    assert all(abs(report['baseline']['audit']['by_query_games'][k]-v)<1e-8 for k,v in expected['audit_baseline_by_k'].items()),'Baseline proxy mismatch'
    assert all(abs(report['full']['audit']['by_query_games'][k]-v)<1e-8 for k,v in expected['audit_selected_by_k'].items()),'Expanded cosine control mismatch'
    return {'results':report,'limitations':['The six original heldout encoders form the local coordinate proxy. LM4 increment is excluded because its all-player training includes these validation IDs.','Four settings fixed before validation; no setting is selected using this audit.','Audit identities were used in earlier work; not a pristine holdout.','No competition upload occurs in this preparation job.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['prepare','verify'],default='prepare');a=p.parse_args();root=a.root.resolve();out=root/'reproduction/tomorrow_five1123'
    if a.mode=='verify':
        manifest=json.loads((out/'candidate_manifest.json').read_text())
        with tempfile.TemporaryDirectory(prefix='replay_',dir=out) as t:records,_=test_candidates(root,Path(t))
        assert {r['file']:r['sha256'] for r in records}=={r['file']:r['sha256'] for r in manifest['candidates']}
        (out/'replay_verification.json').write_text(json.dumps({'verified':True,'candidates':len(records),'check':'All five CSVs regenerated from saved model and frozen inputs; exact SHA256 match'},indent=2));print('FIVE_CANDIDATES_REPLAY_VERIFIED',flush=True);return
    out.mkdir(exist_ok=False);records,hashes=test_candidates(root,out);report=validate(root)
    (out/'validation.json').write_text(json.dumps(report,indent=2));(out/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
    manifest={'prepared_for_quota_day_jst':'2026-11-23','accepted_best_score':.8686865,'source_commit':(root/'reproduction/tomorrow_source_commit1123.txt').read_text().strip(),'candidates':records,'submission_status':'Prepared only; do not duplicate prior artifacts. November24 final submission is manual.'}
    (out/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'REPRODUCE.md').write_text('Restore go-best-20261122 and its frozen parent/policy/cx40 dependencies, plus go-policy-expansion-20261122. Copy the exact source to PROJECT_ROOT/src. Run OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python src/codex_tomorrow_four1123.py --root PROJECT_ROOT. It needs an absent reproduction/tomorrow_five1123 directory. The source verifies the existing expanded candidate and accepted-best control, generates four new fixed hypotheses, validates against the fixed development/audit split and preserves hashes. Then run the same command with --mode verify to regenerate all five CSVs in a temporary directory and require exact SHA matches. No GPU or upload credentials are required. Raw-parent reproduction limitations remain in the accepted best guide.\n')
    print('FIVE_PREPARED',json.dumps(records),flush=True)
if __name__=='__main__':main()
