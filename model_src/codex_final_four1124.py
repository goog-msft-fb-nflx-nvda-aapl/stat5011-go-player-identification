"""Four fixed Task1 experiments for manual final-day submission. Never uploads."""
import argparse, csv, hashlib, json, shutil, tempfile
from pathlib import Path
import numpy as np
from codex_tomorrow_four1123 import norm, z, gaussian, metric

BEST_SHA='eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26'
CONFIGS=[
 {'name':'weight','file':'g24_weight.csv','weight':.15,'hypothesis':'Increase expanded diagonal Gaussian weight from .10 to .15.'},
 {'name':'fullcov','file':'g24_fullcov.csv','weight':.10,'hypothesis':'Global within-player whitening plus .20 candidate covariance adaptation; use Woodbury precision.'},
 {'name':'colour','file':'g24_colour.csv','weight':.10,'hypothesis':'Condition reference means on query colour with three-game shrinkage toward pooled mean.'},
 {'name':'sets','file':'g24_sets.csv','weight':.10,'hypothesis':'Average individual query-game likelihoods instead of likelihood of the query mean.'},
]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def project(x,ck): return norm(((x-ck['mean'])/ck['scale'])@ck['projection'])
def fit_whitener(root,ck,out):
    residuals=[]; ids=[]; hashes={}
    for name in ['policy_residual_train1119','policy_expand1122_0','policy_expand1122_1']:
        folder=root/'reproduction'/name
        for file in ['sample_metadata.npz','policy_residual_features.npz']:
            p=folder/file; hashes[str(p.relative_to(root))]=sha(p)
        meta=np.load(folder/'sample_metadata.npz'); p=meta['player'].reshape(-1,15)
        assert np.all(p==p[:,:1]); ids.extend(p[:,0].tolist())
        e=project(np.load(folder/'policy_residual_features.npz')['residual_mean'].astype(np.float64),ck).reshape(-1,15,128)
        residuals.append((e-e.mean(1,keepdims=True)).reshape(-1,128))
    assert len(ids)==len(set(ids))==6000
    held=np.load(root/'reproduction/policy_residual_pilot1119/sample_metadata.npz')['player']
    assert not set(ids).intersection(held.tolist())
    r=np.concatenate(residuals); c=r.T@r/(len(r)-len(ids)); c=.9*c+.1*np.diag(np.diag(c))+1e-6*np.eye(128)
    w=np.linalg.inv(np.linalg.cholesky(c)).T
    assert np.max(np.abs(w.T@c@w-np.eye(128)))<1e-9
    np.savez(out/'within_player_covariance.npz',covariance=c,whitener=w,train_player_ids=np.array(ids))
    return w,hashes
def fullcov(ref,q,w):
    r=ref@w; qq=q.mean(1)@w; mu=r.mean(1); u=r-mu[:,None,:]; m=r.shape[1]; a=.8; b=.2
    kmat=np.eye(m)[None]+b/(m*a)*(u@u.transpose(0,2,1)); sign,logdet=np.linalg.slogdet(kmat); assert np.all(sign>0)
    ik=np.linalg.inv(kmat)
    proj=np.einsum('qd,cmd->qcm',qq,u)-np.einsum('cd,cmd->cm',mu,u)[None]
    correction=np.einsum('qcm,cmn,qcn->qc',proj,ik,proj,optimize=True)
    dist=(np.sum(qq*qq,1)[:,None]+np.sum(mu*mu,1)[None]-2*qq@mu.T)/a-b/(m*a*a)*correction
    assert dist.min()>-1e-6
    return -.5*np.maximum(dist,0)/(1/q.shape[1]+1/m)-.5*(128*np.log(a)+logdet)[None]
def colour(ref,q,rc,qc):
    mu=ref.mean(1); v=ref.var(1); v=.8*v.mean(0,keepdims=True)+.2*v+1e-5; iv=1/v
    result=np.zeros((len(q),len(ref)))
    for col in [0,1]:
        mask=rc==col; n=mask.sum(1)
        cm=(np.sum(ref*mask[:,:,None],1)+3*mu)/(n[:,None]+3)
        qm=qc==col; qn=qm.sum(1); selected=qn>0
        qq=(q*qm[:,:,None]).sum(1)[selected]/qn[selected,None]
        dist=qq**2@iv.T-2*qq@(cm*iv).T+(cm**2*iv).sum(1)[None]
        result[selected]+=qn[selected,None]/q.shape[1]*(-.5*dist/(1/qn[selected,None]+1/ref.shape[1])-.5*np.log(v).sum(1)[None])
    return result
def sets(ref,q):
    mu=ref.mean(1); v=ref.var(1); v=.8*v.mean(0,keepdims=True)+.2*v+1e-5; iv=1/v
    dist=(q*q).mean(1)@iv.T-2*q.mean(1)@(mu*iv).T+(mu*mu*iv).sum(1)[None]
    return -.5*dist/(1+1/ref.shape[1])-.5*np.log(v).sum(1)[None]
def variants(ref,q,rc,qc,w):
    return {'weight':z(gaussian(ref,q)),'fullcov':z(fullcov(ref,q,w)), 'colour':z(colour(ref,q,rc,qc)), 'sets':z(sets(ref,q))}
def colours(root,pool,d):
    suffix={'pub2':'public','priv2':'private'}[pool]; folder=root/'data/tests1v2/tests'
    cp=folder/f'player_identification_candidates2_{suffix}.csv'; qp=folder/f'player_identification_test2_{suffix}.csv'
    candidates=list(csv.DictReader(cp.open())); queries=list(csv.DictReader(qp.open()))
    ids=sorted(set(r['player_id'] for r in candidates)); assert ids==list(d['player_ids'])
    rc=np.array([[0 if r['color']=='B' else 1 for r in candidates if r['player_id']==pid] for pid in ids]); assert rc.shape==(600,10)
    assert [r['question_id'] for r in queries]==list(d['question_ids'])
    qc=[np.array([0 if r[f'color_{j+1}']=='B' else 1 for j in range(int(r['num_games']))]) for r in queries]
    assert np.array_equal(np.repeat(np.arange(400),[len(c) for c in qc]),d['query_indices'])
    return rc,qc,{str(p.relative_to(root)):sha(p) for p in [cp,qp]}
def test_candidates(root,out,w):
    ckpath=root/'reproduction/policy_expanded_lda1122/policy_lda.npz';ck=np.load(ckpath)
    parent=root/'reproduction/cx38_candidate_pipeline1118'; template=parent/'sub/go19_nover.csv'
    rows=list(csv.reader(template.open(newline=''))); hashes={str(ckpath.relative_to(root)):sha(ckpath),str(template.relative_to(root)):sha(template)}
    predictions={c['name']:{} for c in CONFIGS};predictions['control']={}
    for pool in ['pub2','priv2']:
        dp=root/'reproduction/policy_test1119'/f'{pool}_features.npz'; d=np.load(dp)
        assert np.isfinite(d['features']).all() and (d['counts']>0).all()
        e=project(d['features'],ck); ref=e[:6000].reshape(600,10,-1); qsets=[e[6000:][d['query_indices']==i] for i in range(400)]
        rc,qcols,ch=colours(root,pool,d); hashes.update(ch)
        extras={c['name']:np.empty((400,600)) for c in CONFIGS}
        for k in [1,2,3,5]:
            ix=np.array([i for i,q in enumerate(qsets) if len(q)==k]); assert len(ix)==100
            vv=variants(ref,np.stack([qsets[i] for i in ix]),rc,np.stack([qcols[i] for i in ix]),w)
            for name,s in vv.items(): extras[name][ix]=s
        basepath=root/'reproduction/experiments_1119/runs/cx40_lm4_noaug_all/results'/f'Sfinal_newmember_candidate_{pool}.npy'
        mlppath=root/'reproduction/policy_sweep1120'/f'learned_{pool}.npy'
        base=z(np.load(basepath))+.25*np.load(mlppath)
        score={c['name']:base+c['weight']*extras[c['name']] for c in CONFIGS};score['control']=base+.1*extras['weight']
        for p in [dp,basepath,mlppath]:hashes[str(p.relative_to(root))]=sha(p)
        for name,s in score.items():
            assert np.isfinite(s).all() and s.shape==(400,600)
            for q,order in zip(d['question_ids'],np.argsort(-s,axis=1)[:,:5]):predictions[name][str(q)]=[str(q)]+[str(d['player_ids'][i]) for i in order]
    best=list(csv.reader((root/'reproduction/best1123/g23_greplace.csv').open(newline='')));assert sha(root/'reproduction/best1123/g23_greplace.csv')==BEST_SHA
    previous={sha(p) for p in (root/'reproduction/tomorrow_five1123').glob('*.csv')};records=[];seen=set()
    for config in [dict(name='control',file='best_872195.csv'),*CONFIGS]:
        result=[rows[0]]+[predictions[config['name']].get(row[0],row) for row in rows[1:]]
        assert len(result)==1601 and [r[0] for r in result]==[r[0] for r in rows]
        assert all(len(r)==6 and len(set(r[1:]))==5 for r in result[1:])
        assert all(a==b for a,b in zip(result,best) if a[0].startswith(('pub_q_','priv_q_')))
        p=out/config['file']
        with p.open('w',newline='') as h:csv.writer(h).writerows(result)
        digest=sha(p); assert digest not in seen;seen.add(digest)
        if config['name']=='control':assert digest==BEST_SHA
        else:
            assert digest not in previous, 'Duplicate previous artifact'
            records.append(dict(config,sha256=digest,website_score=None,changed_top1_vs_best=sum(a[1]!=b[1] for a,b in zip(result[1:],best[1:])),changed_rows_vs_best=sum(a!=b for a,b in zip(result[1:],best[1:]))))
    return records,hashes
def validate(root,w):
    import torch
    from torch import nn
    from torch.nn import functional as F
    hroot=root/'reproduction/policy_residual_pilot1119';meta=np.load(hroot/'sample_metadata.npz');H=np.load(hroot/'policy_residual_features.npz')['residual_mean'].astype(np.float64)
    ck=np.load(root/'reproduction/policy_expanded_lda1122/policy_lda.npz');E=project(H,ck).astype(np.float32).reshape(600,15,-1);C=meta['colour'].reshape(600,15)
    cached=np.load(root/'results/HO_meta.npz'); rowof={int(i):j for j,i in enumerate(cached['idx'])}
    base=[norm(np.load(root/'results'/f'HO_{tag}.npy')[[rowof[int(i)] for i in meta['idx']]].astype(np.float32)).reshape(600,15,-1) for tag in ['a2','b1','b2','b3','h4','h5']]
    model=torch.load(root/'reproduction/policy_metric1119/policy_metric.pt',map_location='cpu',weights_only=False)
    net=nn.Sequential(nn.LayerNorm(len(model['mean'])),nn.Linear(len(model['mean']),model['hidden']),nn.GELU(),nn.Dropout(.15),nn.Linear(model['hidden'],model['dimension']));net.load_state_dict(model['net']);net.eval();X=torch.tensor(((H-model['mean'])/model['scale']).astype(np.float32))
    with torch.inference_mode():M=np.concatenate([F.normalize(net(X[i:i+1024]),dim=-1).numpy() for i in range(0,len(X),1024)]).reshape(600,15,-1)
    perm=np.random.default_rng(11202026).permutation(600);groups={'development':np.sort(perm[:300]),'audit':np.sort(perm[300:])};axis=np.arange(600)[:,None]
    results={name:{g:{str(k):[] for k in [1,2,3,5]} for g in groups} for name in ['baseline',*[c['name'] for c in CONFIGS]]}
    for seed in range(5):
        rng=np.random.default_rng(71190+seed);sp=np.stack([rng.permutation(15) for _ in range(600)]);ref,q=sp[:,:10],sp[:,10:]
        for k in [1,2,3,5]:
            original=sum(z(norm(v[axis,q[:,:k]].mean(1))@norm(v[axis,ref].mean(1)).T) for v in base)/6
            original+=.25*z(norm(M[axis,q[:,:k]].mean(1))@norm(M[axis,ref].mean(1)).T)
            vv=variants(E[axis,ref],E[axis,q[:,:k]],C[axis,ref],C[axis,q[:,:k]],w)
            scores={'baseline':original+.1*vv['weight']};scores.update({c['name']:original+c['weight']*vv[c['name']] for c in CONFIGS})
            for name,s in scores.items():
                for g,rr in groups.items():results[name][g][str(k)].append(metric(s,rr))
        print('VALIDATED_SPLIT',seed,flush=True)
    report={name:{g:{'by_query_games':{k:float(np.mean(v)) for k,v in data.items()},'mean':float(np.mean([np.mean(v) for v in data.values()]))} for g,data in gs.items()} for name,gs in results.items()}
    expected=json.loads((root/'reproduction/tomorrow_five1123/validation.json').read_text())['results']['greplace']
    assert all(abs(report['baseline']['audit']['by_query_games'][k]-v)<1e-8 for k,v in expected['audit']['by_query_games'].items()),'Local baseline mismatch'
    return {'results':report,'limitations':['Six original heldout coordinate encoders form the local proxy. LM4 increment excluded because all-player training includes these identities.','Four configurations fixed before audit; no audit-based tuning. Previously reused audit identities are not a pristine holdout.','Same 600-player pool, fixed 300 development / 300 audit query identities, five ten-reference splits and query sizes 1/2/3/5.','Colour variant keeps effective reference count ten for all candidates, an approximation isolating conditional means.','Only updated 600-player pools are changed; original-family predictions stay frozen.']}
def runtime_checks(w):
    rng=np.random.default_rng(20261123);ref=rng.normal(size=(3,10,128));q=rng.normal(size=(4,1,128))
    assert np.allclose(sets(ref,q),gaussian(ref,q),atol=1e-9)
    fast=fullcov(ref,q,w); r=ref@w;qq=q.mean(1)@w
    direct=np.empty_like(fast)
    for i in range(3):
        u=r[i]-r[i].mean(0);cov=.8*np.eye(128)+.2/10*u.T@u;delta=qq-r[i].mean(0)
        direct[:,i]=-.5*np.sum((delta@np.linalg.inv(cov))*delta,1)/1.1-.5*np.linalg.slogdet(cov)[1]
    assert np.allclose(fast,direct,rtol=1e-9,atol=1e-7),'Woodbury direct inverse mismatch'
    return {'single_game_likelihood_equivalence':True,'woodbury_vs_dense_inverse':True,'fullcov_max_abs_error':float(np.max(abs(fast-direct)))}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['prepare','verify'],default='prepare');a=p.parse_args();root=a.root.resolve();out=root/'reproduction/final_four1124'
    if a.mode=='verify':
        manifest=json.loads((out/'candidate_manifest.json').read_text());w=np.load(out/'within_player_covariance.npz')['whitener']
        for rel,digest in json.loads((out/'input_hashes.json').read_text()).items():assert sha(root/rel)==digest,rel
        with tempfile.TemporaryDirectory(prefix='replay_',dir=out) as t:records,_=test_candidates(root,Path(t),w)
        assert {r['file']:r['sha256'] for r in records}=={r['file']:r['sha256'] for r in manifest['candidates']}
        (out/'replay_verification.json').write_text(json.dumps({'verified':True,'four_csv_sha256_matches':True,'accepted_control_sha256':BEST_SHA,'runtime_checks':runtime_checks(w)},indent=2));print('FOUR_REPLAY_VERIFIED',flush=True);return
    out.mkdir(exist_ok=False);ck=np.load(root/'reproduction/policy_expanded_lda1122/policy_lda.npz');w,hashes=fit_whitener(root,ck,out)
    checks=runtime_checks(w);records,inputs=test_candidates(root,out,w);hashes.update(inputs)
    (out/'candidate_manifest.json').write_text(json.dumps({'candidates':records,'preserved_best':{'file':'best_872195.csv','website_score':.8721950,'sha256':BEST_SHA},'submission_mode':'MANUAL ONLY: four experiments first; exact highest-scoring whole CSV last.','deadline_taiwan':'2026-11-24T23:59:59+08:00','deadline_jst':'2026-11-25T00:59:59+09:00','source_sha256':sha(Path(__file__)),'runtime_checks':checks},indent=2))
    (out/'input_hashes.json').write_text(json.dumps(hashes,indent=2));print('FOUR_INFERENCE_COMPLETE',json.dumps(records),flush=True)
    report=validate(root,w);(out/'validation.json').write_text(json.dumps(report,indent=2));shutil.copy2(__file__,out/Path(__file__).name)
    print('FOUR_PREPARED',json.dumps({n:r['audit']['mean'] for n,r in report['results'].items()}),flush=True)
if __name__=='__main__':main()
